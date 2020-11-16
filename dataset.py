import os
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from param import train_param


class MMDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.vids = list(self.data.keys())
        self.transform = transform

    def __getitem__(self, idx):
        vid = self.vids[idx]
        sample = self.data[vid]
        return sample

    def __len__(self):
        return len(self.vids)


def collate_fn(batch_data):
    video = batch_data[0]
    sorted_data = sorted(video.items(), key=lambda item: int(item[0].split('[')[1].split(']')[0]))
    input_ids, masks, visual, lengths, audio, labels = [], [], [], [], [], []
    for i in range(len(sorted_data)):
        t_feats, v_feats, a_feats, label = sorted_data[i][1]
        assert v_feats.shape[0] == a_feats.shape[0]
        input_id = torch.tensor(tokenizer.encode(t_feats))
        mask = torch.ones_like(input_id, dtype=torch.float)
        input_ids.append(input_id)
        masks.append(mask)
        lengths.append(v_feats.shape[0])
        visual.append(torch.tensor(v_feats))
        audio.append(torch.tensor(a_feats))
        labels.append(torch.tensor(label))
    input_ids = pad_sequence(input_ids, batch_first=True)
    masks = pad_sequence(masks, batch_first=True)
    visual = pad_sequence(visual)
    audio = pad_sequence(audio)
    lengths = torch.tensor(lengths)
    labels = torch.tensor(labels)
    return [input_ids, masks, visual, audio, lengths, labels]


def get_loader(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    dataset = MMDataset(data)
    loader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn)
    return loader


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_loader = get_loader(os.path.join(train_param.data_path, 'train.pkl'))
dev_loader = get_loader(os.path.join(train_param.data_path, 'dev.pkl'))
test_loader = get_loader(os.path.join(train_param.data_path, 'test.pkl'))

