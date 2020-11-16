import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.t_encoder = TextEncoder(param.decoder.h_dim)
        self.v_seq = SeqModel(param.v_encoder, param.decoder)
        self.a_seq = SeqModel(param.a_encoder, param.decoder)
        self.linear = nn.Linear(3*param.decoder.h_dim, 1)
        self.weight = Weight(param.weight)
        self.video = Video(param.decoder.h_dim, self.weight.w_u)

    def forward(self, t_inputs, mask, v_inputs, a_inputs, lengths):
        t_utter, d_inputs = self.t_encoder(t_inputs, mask)
        d_lengths = torch.sum(mask, dim=1, dtype=torch.int) - 1
        v_outputs, v_utter = self.v_seq(v_inputs, lengths, d_inputs, d_lengths)
        a_outputs, a_utter = self.a_seq(a_inputs, lengths, d_inputs, d_lengths)
        utter = torch.cat([t_utter, v_utter, a_utter], -1)
        y_hat = self.linear(utter)
        y_hat = self.video(utter, y_hat).squeeze(1)
        return v_outputs, a_outputs, y_hat, self.weight.w_s


class SeqModel(nn.Module):
    def __init__(self,  encoder_param, decoder_param):
        super().__init__()
        self.encoder = Encoder(*encoder_param)
        self.decoder = Decoder(*decoder_param)

    def forward(self, e_inputs, e_lengths, d_inputs, d_lengths):
        e_outputs, hidden, utter = self.encoder(e_inputs, e_lengths)
        d_outputs = self.decoder(d_inputs, d_lengths, hidden, e_outputs)
        return d_outputs, utter


class Encoder(nn.Module):
    def __init__(self, i_dim, h_dim, layers, dropout, bi):
        super(Encoder, self).__init__()
        self.h_dim = h_dim
        self.layers = layers
        self.bi = bi
        self.gru = nn.GRU(i_dim, h_dim, num_layers=layers, dropout=dropout, bidirectional=bool(bi))

    def forward(self, inputs, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = torch.split(outputs, self.h_dim, dim=2)
        outputs = sum(outputs) / len(outputs)
        hidden = torch.split(hidden.view(self.layers, -1, hidden.size(1), hidden.size(2)), 1, dim=1)
        hidden = torch.squeeze(sum(hidden) / len(hidden), 1)
        utter = hidden[-1]
        return outputs, hidden, utter


class Decoder(nn.Module):
    def __init__(self, i_dim, h_dim, o_dim, layers, dropout):
        super(Decoder, self).__init__()
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.layers = layers
        self.dropout = dropout
        self.gru = nn.GRU(i_dim, h_dim, num_layers=layers, dropout=dropout)
        self.linear_1 = nn.Linear(2*h_dim, h_dim)
        self.act_1 = nn.Tanh()
        self.linear_2 = nn.Linear(h_dim, o_dim)
        self.act_2 = nn.Softmax(dim=2)

    def forward(self, inputs, lengths, hidden, attend):
        packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths, enforce_sorted=False)
        outputs, _ = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        query = outputs.transpose(0, 1)
        key = attend.transpose(0, 1).transpose(1, 2)
        value = attend.transpose(0, 1)
        attn = attention(query, key, value)
        outputs = torch.cat((outputs, attn), 2)
        outputs = self.act_1(self.linear_1(outputs))
        outputs = self.act_2(self.linear_2(outputs))
        return outputs


class TextEncoder(nn.Module):
    def __init__(self, h_dim):
        super(TextEncoder, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config)
        self.linear = nn.Linear(self.config.hidden_size, h_dim)
        self.embedding = self.model.embeddings.word_embeddings

    def forward(self, inputs, mask):
        outputs = self.model(input_ids=inputs, attention_mask=mask)
        encoded = outputs[1]
        encoded = self.linear(encoded)
        embeddings = self.embedding(inputs)
        embeddings = embeddings.transpose(0, 1)[:-1]
        return encoded, embeddings


class Video(nn.Module):
    def __init__(self, h_dim, w_u):
        super().__init__()
        self.linear = nn.Linear(3*h_dim, 1)
        self.weight = w_u

    def forward(self, utter, y_hat):
        video = torch.mean(utter, dim=0)
        y_v = self.linear(video)
        y_hat = y_v + self.weight*y_hat
        return y_hat


class Weight(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.w_u = nn.Parameter(torch.tensor(weight.w_u), requires_grad=bool(weight.train))
        self.w_s = nn.Parameter(torch.tensor(weight.w_s), requires_grad=bool(weight.train))


def attention(query, key, value):
    if query.dim() == 3:
        score = F.softmax(torch.bmm(query, key), dim=2)
        attn = torch.bmm(score, value)
        return attn.transpose(0, 1)
    else:
        score = F.softmax(torch.matmul(query, key), dim=1)
        attn = torch.matmul(score, value)
        return attn




