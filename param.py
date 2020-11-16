import torch.nn as nn
from collections import namedtuple


class ModelParam(object):
    def __init__(self, params):
        super().__init__()
        self.lr = params[0]
        self.v_encoder = params[1]
        self.a_encoder = params[2]
        self.decoder = params[3]
        self.weight = params[4]

    def state_dict(self):
        return self


TrainParam = namedtuple('TrainParam', ['data_path', 'save_path',
                                       'lr', 'epoch', 'patience', 'device', 'interval', 'grad_norm', 'seed'])
EncoderParam = namedtuple('EncoderParam', ['i_dim', 'h_dim', 'layers', 'dropout', 'bi'])
DecoderParam = namedtuple('DecoderParam', ['i_dim', 'h_dim', 'o_dim', 'layers', 'dropout'])
LoadParam = namedtuple('LoadParam', ['ckpt_path', 'res_path', 'columns', 'device'])
WeightParam = namedtuple('WeightParam', ['w_u', 'w_s', 'train'])

train_param = TrainParam('/data/path',
                         '/save/checkpoint/path',
                         1e-5, 60, 5, 0, 10, 1.0, 7)
v_encoder = EncoderParam(47, 256, 2, 0.5, 1)
a_encoder = EncoderParam(74, 256, 2, 0.5, 1)
decoder = DecoderParam(768, 256, 30522, 2, 0.5)
weight = WeightParam(0.1, 0.1, 0)
model_param = ModelParam((train_param.lr, v_encoder, a_encoder, decoder, weight))
columns = ['lr', 'h_dim', 'layers', 'dropout', 'bi', 'w_u', 'w_s', 'train',
           'acc2', 'f1', 'mae', 'corr', 'acc7', 'acc5']
load_param = LoadParam('/load/checkpoint/path',
                       '/load/result/path',
                       columns,
                       0)
