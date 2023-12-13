from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

from .module_encoder import TfModel, TextConfig, VisualConfig, AudioConfig
from .until_module import PreTrainedModel, LayerNorm
from .until_module import getBinaryTensor, CTCModule, MLLinear, MLAttention
import warnings
from .losses import *

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)



class CARATPreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, text_config, visual_config, audio_config,*inputs, **kwargs):
        # utilize bert config as base config
        super(CARATPreTrainedModel, self).__init__(visual_config)
        self.text_config = text_config
        self.visual_config = visual_config
        self.audio_config = audio_config
        self.visual = None
        self.audio = None
        self.text = None

    
    @classmethod
    def from_pretrained(cls, text_model_name, visual_model_name, audio_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0
        text_config, _= TextConfig.get_config(text_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        audio_config, _ = AudioConfig.get_config(audio_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        model = cls(text_config, visual_config, audio_config, *inputs, **kwargs)
        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        return model


class Normalize(nn.Module):
    def __init__(self, dim):
        super(Normalize, self).__init__()
        self.norm2d = LayerNorm(dim)

    def forward(self, inputs):
        inputs = torch.as_tensor(inputs).float()
        inputs = inputs.view(-1, inputs.shape[-2], inputs.shape[-1])
        output = self.norm2d(inputs)
        return output


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config
    
class CARAT(CARATPreTrainedModel):
    def __init__(self, text_config, visual_config, audio_config, task_config):
        super(CARAT, self).__init__(text_config, visual_config, audio_config)
        self.task_config = task_config
        self.num_classes = task_config.num_classes
        self.aligned = task_config.aligned
        self.proto_m = task_config.proto_m

        text_config = update_attr("text_config", text_config, "num_hidden_layers",
                                  self.task_config, "text_num_hidden_layers")
        self.text = TfModel(text_config)
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = TfModel(visual_config)
        audio_config = update_attr("audio_config", audio_config, "num_hidden_layers",
                                   self.task_config, "audio_num_hidden_layers")
        self.audio = TfModel(audio_config)

        self.text_norm = Normalize(task_config.text_dim)
        self.visual_norm = Normalize(task_config.video_dim)
        self.audio_norm = Normalize(task_config.audio_dim)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.criterion_cl = SupConLoss()

        self.apply(self.init_weights)

        self.text_attention = MLAttention(self.num_classes, task_config.hidden_size)
        self.visual_attention = MLAttention(self.num_classes, task_config.hidden_size)
        self.audio_attention = MLAttention(self.num_classes, task_config.hidden_size)

        self.proj_text = MLLinear([task_config.hidden_size, task_config.hidden_size//2], task_config.proj_size)
        self.proj_visual = MLLinear([task_config.hidden_size, task_config.hidden_size//2], task_config.proj_size)
        self.proj_audio = MLLinear([task_config.hidden_size, task_config.hidden_size//2], task_config.proj_size)

        self.de_proj_text = MLLinear([task_config.proj_size, task_config.hidden_size//2], task_config.hidden_size)
        self.de_proj_visual = MLLinear([task_config.proj_size, task_config.hidden_size//2], task_config.hidden_size)
        self.de_proj_audio = MLLinear([task_config.proj_size, task_config.hidden_size//2], task_config.hidden_size)


        self.tv2a = MLLinear([task_config.hidden_size * 3], task_config.hidden_size)
        self.ta2v = MLLinear([task_config.hidden_size * 3], task_config.hidden_size)
        self.va2t = MLLinear([task_config.hidden_size * 3], task_config.hidden_size)
        self.max_pool = nn.MaxPool1d(3)

        self.agg = MLLinear([task_config.hidden_size * self.num_classes, task_config.hidden_size], self.num_classes)

        self.text_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.text_clf_weight, a=math.sqrt(5))
        self.visual_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.visual_clf_weight, a=math.sqrt(5))
        self.audio_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.audio_clf_weight, a=math.sqrt(5))

        self.sigmoid = nn.Sigmoid()

        self.register_buffer('text_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('text_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('visual_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('visual_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('audio_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('audio_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))

        self.register_buffer('queue', torch.randn(task_config.moco_queue, task_config.proj_size))
        self.register_buffer("queue_label", torch.randn(task_config.moco_queue, 1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)

        if not self.aligned:
            self.a2t_ctc = CTCModule(task_config.audio_dim, 50 if task_config.unaligned_mask_same_length else 500)
            self.v2t_ctc = CTCModule(task_config.video_dim, 50 if task_config.unaligned_mask_same_length else 500)

    def dequeue_and_enqueue(self, feats, labels):
        batch_size = feats.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size >= self.task_config.moco_queue:
            self.queue[ptr:,:] = feats[:self.task_config.moco_queue-ptr,:]
            self.queue[:batch_size - self.task_config.moco_queue + ptr,:] = feats[self.task_config.moco_queue-ptr:,:]
            self.queue_label[ptr:, :] = labels[:self.task_config.moco_queue - ptr, :]
            self.queue_label[:batch_size - self.task_config.moco_queue + ptr, :] = labels[self.task_config.moco_queue - ptr:,
                                                                             :]
        else:
            self.queue[ptr:ptr+batch_size, :] = feats
            self.queue_label[ptr:ptr + batch_size, :] = labels
        ptr = (ptr + batch_size) % self.task_config.moco_queue  # move pointer
        self.queue_ptr[0] = ptr


    def get_text_visual_audio_output(self, text, text_mask, visual, visual_mask, audio, audio_mask):
        text_layers, text_pooled_output = self.text(text, text_mask, output_all_encoded_layers=True)
        text_output = text_layers[-1]
        visual_layers, visual_pooled_output = self.visual(visual, visual_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]
        audio_layers, audio_pooled_output = self.audio(audio, audio_mask, output_all_encoded_layers=True)
        audio_output = audio_layers[-1]
        return text_output, visual_output, audio_output

    def get_cl_labels(self, labels, times = 1):
        text_labels = torch.zeros_like(labels) + labels
        visual_labels = torch.zeros_like(labels) + labels
        audio_labels = torch.zeros_like(labels) + labels

        text_cl_labels = torch.zeros_like(text_labels, dtype=torch.long)
        visual_cl_labels = torch.zeros_like(visual_labels, dtype=torch.long)
        audio_cl_labels = torch.zeros_like(audio_labels, dtype=torch.long)

        example_idx, label_idx = torch.where(text_labels >= 0.5)
        text_cl_labels[example_idx, label_idx] = label_idx
        example_idx, label_idx = torch.where(text_labels < 0.5)
        text_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 1

        example_idx, label_idx = torch.where(visual_labels >= 0.5)
        visual_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 2
        example_idx, label_idx = torch.where(visual_labels < 0.5)
        visual_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 3

        example_idx, label_idx = torch.where(audio_labels >= 0.5)
        audio_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 4
        example_idx, label_idx = torch.where(audio_labels < 0.5)
        audio_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 5

        cl_labels = torch.stack([text_cl_labels, visual_cl_labels, audio_cl_labels], dim=1)

        cl_labels = cl_labels.to(torch.int)
        if times > 1:
            final_cl_labels = torch.cat([cl_labels, cl_labels], dim=1)
            for i in range(2, times):
                final_cl_labels = torch.cat([final_cl_labels, cl_labels], dim=1)
        else:
            final_cl_labels = cl_labels
        return final_cl_labels

    def get_cl_mask(self, cl_labels, batch_size):
        mask = torch.eq(cl_labels[:batch_size], cl_labels.T).float()
        neg_mask = torch.ones_like(mask)
        return mask, neg_mask

    def update_protos(self, pos_protos, neg_protos, feats, gt_labels):
        b, c = gt_labels.shape[0], gt_labels.shape[1]
        for i in range(b):
            for j in range(c):
                if gt_labels[i][j] == 1:
                    pos_protos[j] = pos_protos[j] * self.proto_m + (1 - self.proto_m) * feats[i][j]
                else:
                    neg_protos[j] = neg_protos[j] * self.proto_m + (1 - self.proto_m) * feats[i][j]

    def forward(self, text, text_mask, visual, visual_mask, audio, audio_mask,
                label_input, label_mask, groundTruth_labels=None, training=True):
        text = self.text_norm(text)
        visual = self.visual_norm(visual)
        audio = self.audio_norm(audio)
        if self.aligned == False:
            visual, v2t_position = self.v2t_ctc(visual)
            audio, a2t_position = self.a2t_ctc(audio)
        text_output, visual_output, audio_output = self.get_text_visual_audio_output(text, text_mask, visual,
                                                                                     visual_mask, audio,
                                                                                     audio_mask)  # [B, L, D]
        text_lsr, text_attention = self.text_attention(text_output, (1 - text_mask).type(torch.bool))
        visual_lsr, visual_attention = self.visual_attention(visual_output, (1 - visual_mask).type(torch.bool))
        audio_lsr, audio_attention = self.audio_attention(audio_output, (1 - audio_mask).type(torch.bool))

        latent_text = self.proj_text(text_lsr)
        latent_visual = self.proj_visual(visual_lsr)
        latent_audio = self.proj_audio(audio_lsr)
        recon_text = self.de_proj_text(latent_text)
        recon_visual = self.de_proj_visual(latent_visual)
        recon_audio = self.de_proj_audio(latent_audio)


        text_n = F.normalize(latent_text, p=2, dim=-1)
        visual_n = F.normalize(latent_visual, p=2, dim=-1)
        audio_n = F.normalize(latent_audio, p=2, dim=-1)
        text_protos = torch.stack([self.text_pos_protos, self.text_neg_protos])
        visual_protos = torch.stack([self.visual_pos_protos, self.visual_neg_protos])
        audio_protos = torch.stack([self.audio_pos_protos, self.audio_neg_protos])
        text_sim = torch.einsum('bld,nld->bln', text_n, text_protos)
        visual_sim = torch.einsum('bld,nld->bln', visual_n, visual_protos)
        audio_sim = torch.einsum('bld,nld->bln', audio_n, audio_protos)
        text_sim = torch.softmax(text_sim, dim=-1)
        visual_sim = torch.softmax(visual_sim, dim=-1)
        audio_sim = torch.softmax(audio_sim, dim=-1)
        if not training:
            text_pos_sim, text_neg_sim = text_sim[:, :, 0], text_sim[:, :, 1]
            visual_pos_sim, visual_neg_sim = visual_sim[:, :, 0], visual_sim[:, :, 1]
            audio_pos_sim, audio_neg_sim = audio_sim[:, :, 0], audio_sim[:, :, 1]
            text_pos_mask = (text_pos_sim > text_neg_sim).to(torch.float)
            text_neg_mask = 1 - text_pos_mask
            visual_pos_mask = (visual_pos_sim > visual_neg_sim).to(torch.float)
            visual_neg_mask = 1 - visual_pos_mask
            audio_pos_mask = (audio_pos_sim > audio_neg_sim).to(torch.float)
            audio_neg_mask = 1 - audio_pos_mask
            text_latent_padding = text_pos_mask.unsqueeze(-1) * self.text_pos_protos.unsqueeze(0) + \
                                  text_neg_mask.unsqueeze(-1) * self.text_neg_protos.unsqueeze(0)
            visual_latent_padding = visual_pos_mask.unsqueeze(-1) * self.visual_pos_protos.unsqueeze(0) + \
                                  visual_neg_mask.unsqueeze(-1) * self.visual_neg_protos.unsqueeze(0)
            audio_latent_padding = audio_pos_mask.unsqueeze(-1) * self.audio_pos_protos.unsqueeze(0) + \
                                  audio_neg_mask.unsqueeze(-1) * self.audio_neg_protos.unsqueeze(0)
        else:
            text_latent_padding = torch.einsum('bln,nld->bld', text_sim, text_protos)
            visual_latent_padding = torch.einsum('bln,nld->bld', visual_sim, visual_protos)
            audio_latent_padding = torch.einsum('bln,nld->bld', audio_sim, audio_protos)
        text_padding = self.de_proj_text(text_latent_padding)
        visual_padding = self.de_proj_visual(visual_latent_padding)
        audio_padding = self.de_proj_audio(audio_latent_padding)



        audio_aug = self.tv2a(torch.cat([recon_text, recon_visual, audio_padding], dim=-1))
        visual_aug = self.ta2v(torch.cat([recon_text, visual_padding, recon_audio], dim=-1))
        text_aug = self.va2t(torch.cat([text_padding, recon_visual, recon_audio], dim=-1))
        text_clf_out_3 = torch.einsum('bld,ld->bl', text_aug, self.text_clf_weight)
        visual_clf_out_3 = torch.einsum('bld,ld->bl', visual_aug, self.visual_clf_weight)
        audio_clf_out_3 = torch.einsum('bld,ld->bl', audio_aug, self.audio_clf_weight)

        audio_beta = self.tv2a(torch.cat([text_aug, visual_aug, audio_aug], dim=-1))
        visual_beta = self.ta2v(torch.cat([text_aug, visual_aug, audio_aug], dim=-1))
        text_beta = self.va2t(torch.cat([text_aug, visual_aug, audio_aug], dim=-1))
        text_clf_out_4 = torch.einsum('bld,ld->bl', text_beta, self.text_clf_weight)
        visual_clf_out_4 = torch.einsum('bld,ld->bl', visual_beta, self.visual_clf_weight)
        audio_clf_out_4 = torch.einsum('bld,ld->bl', audio_beta, self.audio_clf_weight)

        text_clf_out_1 = torch.einsum('bld,ld->bl', text_lsr, self.text_clf_weight)
        visual_clf_out_1 = torch.einsum('bld,ld->bl', visual_lsr, self.visual_clf_weight)
        audio_clf_out_1 = torch.einsum('bld,ld->bl', audio_lsr, self.audio_clf_weight)


        if training:
            latent_aug_text = self.proj_text(text_aug)
            latent_aug_visual = self.proj_visual(visual_aug)
            latent_aug_audio = self.proj_audio(audio_aug)
            latent_beta_text = self.proj_text(text_beta)
            latent_beta_visual = self.proj_visual(visual_beta)
            latent_beta_audio = self.proj_audio(audio_beta)
            total_proj = torch.stack([latent_text, latent_visual, latent_audio,
                                      latent_aug_text, latent_aug_visual, latent_aug_audio,
                                      latent_beta_text, latent_beta_visual, latent_beta_audio], dim=1)
            label_time = 3

            total_proj = total_proj.view(-1, total_proj.shape[-1])
            total_proj = F.normalize(total_proj, dim=-1)
            cl_labels = self.get_cl_labels(groundTruth_labels, times=label_time).view(-1).unsqueeze(-1)
            text_norm = F.normalize(latent_text.data, dim=-1)
            visual_norm = F.normalize(latent_visual.data, dim=-1)
            audio_norm = F.normalize(latent_audio.data, dim=-1)
            cl_feats = torch.cat((total_proj, self.queue.clone().detach()), dim=0)
            total_cl_labels = torch.cat((cl_labels, self.queue_label.clone().detach()), dim=0)
            batch_size = cl_feats.shape[0]
            cl_mask, cl_neg_mask = self.get_cl_mask(total_cl_labels, batch_size)
            cl_loss = self.criterion_cl(cl_feats, cl_mask, cl_neg_mask, batch_size)
            self.dequeue_and_enqueue(total_proj, cl_labels)
            self.update_protos(self.text_pos_protos, self.text_neg_protos, text_norm, groundTruth_labels)
            self.update_protos(self.visual_pos_protos, self.visual_neg_protos, visual_norm, groundTruth_labels)
            self.update_protos(self.audio_pos_protos, self.audio_neg_protos, audio_norm, groundTruth_labels)
        # predict_scores_text4 = self.sigmoid(text_clf_out_4)
        # predict_scores_visual4 = self.sigmoid(visual_clf_out_4)
        # predict_scores_audio4 = self.sigmoid(audio_clf_out_4)
        # # predict_scores_mean = (predict_scores_text4 + predict_scores_visual4 + predict_scores_audio4) / 3


        clf_out_1 = torch.stack([text_clf_out_1, visual_clf_out_1, audio_clf_out_1], dim=-1)
        clf_out_1 = self.max_pool(clf_out_1).squeeze(-1)
        clf_out_3 = torch.stack([text_clf_out_3, visual_clf_out_3, audio_clf_out_3], dim=-1)
        clf_out_3 = self.max_pool(clf_out_3).squeeze(-1)
        clf_out_4 = torch.stack([text_clf_out_4, visual_clf_out_4, audio_clf_out_4], dim=-1)
        clf_out_4 = self.max_pool(clf_out_4).squeeze(-1)
        predict_scores_clf4 = self.sigmoid(clf_out_4)
        # predict_labels_clf4 = getBinaryTensor(predict_scores_clf4, boundary=self.task_config.binary_threshold)
        # max_scores = torch.stack([predict_scores_clf4, predict_scores_clf4, ])
        # max_labels = torch.stack([predict_labels_clf4, predict_labels_clf4, ])
        # predict_labels = torch.stack([max_labels, max_labels], dim=0)
        # predict_scores = torch.stack([max_scores, max_scores], dim=0)


        total_aug = torch.stack([text_beta, visual_beta, audio_beta], dim=1)
        agg_out = self.agg(total_aug.view(total_aug.shape[0], total_aug.shape[1], -1))
        agg_scores = self.sigmoid(agg_out)
        predict_agg_scores = torch.mean(agg_scores, dim=1)
        # predict_agg_labels = getBinaryTensor(predict_agg_scores, boundary=self.task_config.binary_threshold)
        # predict_labels = torch.cat([predict_labels, predict_agg_labels.unsqueeze(0)], dim=0)
        # predict_scores = torch.cat([predict_scores, predict_agg_scores.unsqueeze(0)], dim=0)
        # predict_agg_scores_mean = (predict_agg_scores + predict_scores_mean) / 2
        # predict_agg_labels_mean = getBinaryTensor(predict_agg_scores_mean, boundary=self.task_config.binary_threshold)
        # predict_labels = torch.cat([predict_labels, predict_agg_labels_mean.unsqueeze(0)], dim=0)
        # predict_scores = torch.cat([predict_scores, predict_agg_scores_mean.unsqueeze(0)], dim=0)

        predict_final_scores_mean = (predict_agg_scores + predict_scores_clf4) / 2
        predict_final_labels_mean = getBinaryTensor(predict_final_scores_mean,
                                                  boundary=self.task_config.binary_threshold)
        predict_scores = predict_final_scores_mean
        predict_labels = predict_final_labels_mean
        # predict_scores = torch.cat([predict_scores, predict_final_scores_mean.unsqueeze(0)], dim=0)

        if training:
            total_aug_clf_loss = self.bce_loss(agg_out, groundTruth_labels.unsqueeze(-2).repeat(1, 3, 1))
            shuffle_sample_idx = torch.zeros(self.num_classes, total_aug.shape[1], total_aug.shape[0], dtype=torch.long)
            for l in range(self.num_classes):
                for m in range(total_aug.shape[1]):
                    one_idx = np.random.permutation(total_aug.shape[0])
                    shuffle_sample_idx[l][m] += one_idx
            shuffle_sample_idx = shuffle_sample_idx.permute(2, 1, 0)

            shuffle_modality_idx = torch.zeros(self.num_classes, total_aug.shape[0], total_aug.shape[1],
                                               dtype=torch.long)
            for l in range(self.num_classes):
                for s in range(total_aug.shape[0]):
                    one_idx = np.random.permutation(total_aug.shape[1])
                    shuffle_modality_idx[l][s] += one_idx
            shuffle_modality_idx = shuffle_modality_idx.permute(1, 2, 0)

            label_idx = torch.zeros(total_aug.shape[0], total_aug.shape[1], self.num_classes) + torch.tensor(
                list(range(self.num_classes)))
            label_idx = label_idx.to(torch.long)
            shuffle_total_aug = total_aug[shuffle_sample_idx, shuffle_modality_idx, label_idx]
            shuffle_aug_out = self.agg(shuffle_total_aug.view(total_aug.shape[0], total_aug.shape[1], -1))
            shuffle_gt_labels = groundTruth_labels.unsqueeze(-2).repeat(1, 3, 1)[shuffle_sample_idx, shuffle_modality_idx, label_idx]
            shuffle_aug_clf_loss = self.bce_loss(shuffle_aug_out, shuffle_gt_labels)

        if training:
            all_loss = 0
            clf_loss = self.bce_loss(clf_out_1, groundTruth_labels) * self.task_config.lsr_clf_weight
            clf_loss += self.bce_loss(clf_out_3, groundTruth_labels) * self.task_config.aug_clf_weight
            clf_loss += self.bce_loss(clf_out_4, groundTruth_labels)
            all_loss += clf_loss

            all_loss += cl_loss * self.task_config.cl_weight

            aug_mse_loss = self.mse_loss(text_aug, text_lsr) + self.mse_loss(visual_aug, visual_lsr)\
                           + self.mse_loss(audio_aug, audio_lsr)
            beta_mse_loss = self.mse_loss(text_beta, text_lsr) + self.mse_loss(visual_beta, visual_lsr)\
                            + self.mse_loss(audio_beta, audio_lsr)
            recon_mse_loss = self.mse_loss(recon_text, text_lsr) + self.mse_loss(recon_visual, visual_lsr) \
                             + self.mse_loss(recon_audio, audio_lsr)
            all_loss += recon_mse_loss * self.task_config.recon_mse_weight\
                        + aug_mse_loss * self.task_config.aug_mse_weight + beta_mse_loss * self.task_config.beta_mse_weight

            all_loss += total_aug_clf_loss * self.task_config.total_aug_clf_weight
            all_loss += shuffle_aug_clf_loss * self.task_config.shuffle_aug_clf_weight
            return all_loss, predict_labels, groundTruth_labels, predict_scores
        else:

            return predict_labels, groundTruth_labels, predict_scores


