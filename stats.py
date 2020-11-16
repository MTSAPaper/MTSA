import os
from collections import OrderedDict

import torch
import pickle
import pandas as pd

from model import Model
from metric import metric
from param import load_param
from dataset import test_loader
from handler import create_evaluator, form


def load(ckpt_path, columns):
    checkpoints = os.listdir(ckpt_path)
    stats = OrderedDict()
    device = torch.device(f'cuda:{load_param.device}')
    for k in columns:
        stats[k] = list()
    for ckpt in checkpoints:
        checkpoint = torch.load(os.path.join(ckpt_path, ckpt))
        ckpt_param = checkpoint['param']
        model = Model(ckpt_param)
        model.load_state_dict(checkpoint['model'])
        test_evaluator = create_evaluator(model, metric, device)
        test_evaluator.run(test_loader)
        metrics = form(test_evaluator.state.metrics)
        record(stats, ckpt_param, metrics)
    stats = pd.DataFrame(stats, index=checkpoints)
    return stats


def record(stats, ckpt_param, metrics):
    stats['lr'].append(ckpt_param.lr)
    stats['h_dim'].append(ckpt_param.v_encoder.h_dim)
    stats['layers'].append(ckpt_param.v_encoder.layers)
    stats['dropout'].append(ckpt_param.v_encoder.dropout)
    stats['bi'].append(ckpt_param.v_encoder.bi)
    stats['w_u'].append(ckpt_param.weight.w_u)
    stats['w_s'].append(ckpt_param.weight.w_s)
    stats['train'].append(ckpt_param.weight.train)
    stats['acc2'].append(metrics['acc2'])
    stats['f1'].append(metrics['f1'])
    stats['mae'].append(metrics['mae'])
    stats['corr'].append(metrics['corr'])
    stats['acc7'].append(metrics['acc7'])
    stats['acc5'].append(metrics['acc5'])


def save(stats, res_file):
    with open(res_file, 'wb') as f:
        pickle.dump(stats, f)


def write(ckpt_path, columns, res_file):
    stats = load(ckpt_path, columns)
    save(stats, res_file)
    return stats


def read(res_file):
    with open(res_file, 'rb') as f:
        stats = pickle.load(f)
    return stats


if __name__ == '__main__':
    checkpoint_path = load_param.ckpt_path
    result_path = load_param.res_path
    result_file = os.path.join(result_path, 'results.pkl')
    if not os.path.isdir(checkpoint_path):
        raise ValueError('Checkpoint path is not a directory.')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isfile(result_file):
        res = write(checkpoint_path, load_param.columns, result_file)
    else:
        res = read(result_file)
        if sorted(res.index) != sorted(os.listdir(checkpoint_path)):
            res = write(checkpoint_path, load_param.columns, result_file)
    print(res)
