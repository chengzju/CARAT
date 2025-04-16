import os
import torch
import json
import numpy as np
import pickle
import collections

file_path = os.path.join('./data', 'train_valid_test.pt')
all_data = torch.load(file_path)

file_name_1 = './data/AIS10_norm-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed.pkl'
file_name_2 = './data/Asent_avg_wav2vec_zh-Vsent_avg_affectdenseface-Lsent_avg_robert_base_wwm_chinese.pkl'
file_name_3 = './data/Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_avg_robert_base_wwm_chinese.pkl'
file_name_4 = './data/Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed.pkl'

with open(file_name_4, 'rb') as f:
    data = pickle.load(f)
    print(type(data), len(data))

for id, i in enumerate(data):
    print(id, type(i))
    if isinstance(i, list):
        print(len(i), i[0])
    elif isinstance(i, collections.OrderedDict):
        print(len(i.keys()))
        for k,v in i.items():
            print(k,type(v))
            if isinstance(v, np.ndarray):
                print(v.shape)
                # if id == 6:
                #     print(v)
            elif isinstance(v, list):
                print(len(v), v)
            else:
                print(v)
            break
    print('------------------')

labels = data[2]
dialogues = data[0]
max_len = 0
total_len = 0
cnt = 0

max_label_cnt = 0
total_label_cnt = 0

def get_sum(one_list):
    cnt = 0
    for o in one_list:
        cnt += o
    return cnt

print(0)
for k, v in labels.items():
    ins_len = len(v)
    max_len = max(max_len, ins_len)
    total_len += ins_len
    cnt += 1
    one_label = [0, 0, 0, 0, 0, 0, 0]
    for o in v:
        one_label[o] = 1
    one_label_cnt = get_sum(one_label)
    max_label_cnt = max(max_label_cnt, one_label_cnt)
    total_label_cnt += one_label_cnt
print(cnt)
print(max_len, total_len/cnt)
print(max_label_cnt, total_label_cnt/cnt)

labels = data[2]
t_dist = data[3]
v_dist = data[4]
a_dist = data[5]
train_idx = data[7]
val_idx = data[8]
text_idx = data[9]

total_len = 100
final_len = 60
t_dim = 768
v_dim = 1024
a_dim = 342


def get_one_data(train_idx):
    train_data = {}
    src_v = []
    src_t = []
    src_a = []
    len_list = []
    tgt = []
    for id in train_idx:
        one_label = labels[id]
        cur_len = len(one_label)
        len_list.append(cur_len)
        one_label_list = [2]
        for l in one_label:
            ll = l + 4
            if ll not in one_label_list:
                one_label_list.append(ll)
        one_label_list.append(3)
        tgt.append(one_label_list)
        t_data = torch.tensor(t_dist[id])
        v_data = torch.tensor(v_dist[id])
        a_data = torch.tensor(a_dist[id])
        t_padding = torch.zeros(total_len - cur_len, t_dim)
        v_padding = torch.zeros(total_len - cur_len, v_dim)
        a_padding = torch.zeros(total_len - cur_len, a_dim)
        t_tensor = torch.cat([t_data, t_padding], dim=0)
        v_tensor = torch.cat([v_data, v_padding], dim=0)
        a_tensor = torch.cat([a_data, a_padding], dim=0)
        t_tensor = t_tensor[:final_len].numpy()
        v_tensor = v_tensor[:final_len].numpy()
        a_tensor = a_tensor[:final_len].numpy()
        src_v.append(v_tensor)
        src_a.append(a_tensor)
        src_t.append(t_tensor)
    train_data['src-visual'] = src_v
    train_data['src-text'] = src_t
    train_data['src-audio'] = src_a
    train_data['len_list'] = len_list
    train_data['tgt'] = tgt

    return train_data

train_data = get_one_data(train_idx)
val_data = get_one_data(val_idx)
test_data = get_one_data(text_idx)

final_data = {}
final_data['train'] = train_data
final_data['valid'] = val_data
final_data['test'] = test_data
info_dict = {}
info_dict['tgt'] = {'<s>': 2, '</s>': 3, '<unk>': 1, '<blank>': 0, 'Happy':4, 'Neutral':5, 'Sad':6, 'Disgust':7, 'Anger': 8, 'Fear': 9, 'Surprise':10}
info_dict['src'] = all_data['dict']['src']
final_data['dict'] = info_dict
final_data['settings'] = all_data['settings']
torch.save(final_data, './data/m3ed_data_{}.pt'.format(final_len))

