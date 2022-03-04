import random

import numpy.linalg
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from traj_generator import load_dataset
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class EncoderRNN(nn.Module):
    def __init__(self, args, device):
        super(EncoderRNN, self).__init__()
        self.args = args
        self.device = device
        self.rnn = nn.GRU(self.args.encoder_input_dim, self.args.encoder_hidden_dim, dropout=self.args.encoder_dropout, bidirectional=True)
        self.fc = nn.Linear(self.args.encoder_hidden_dim * 2, self.args.decoder_hidden_dim)
        # self.dropout = nn.Dropout(args.encoder_dropout)

    def forward(self, input_seq):
        input_seq = input_seq.unsqueeze(dim=1)
        output, hidden = self.rnn(input_seq)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, self.args.batch_size, self.args.encoder_hidden_dim, device=self.device)


class Attention(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.attn = nn.Linear((self.args.encoder_hidden_dim * 2) + self.args.decoder_hidden_dim,
                              self.args.decoder_hidden_dim)
        self.v = nn.Linear(self.args.decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # print(hidden.shape)
        # print(encoder_outputs.shape)

        # batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # hidden = hidden.permute(1, 0, -1)
        # print(hidden.shape)
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # print(hidden.shape)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class AttnDecoderRNN(nn.Module):
    def __init__(self, attention, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.output_dim = args.decoder_output_dim
        self.attention = attention

        self.rnn = nn.GRU((args.encoder_hidden_dim * 2) + self.args.encoder_input_dim, args.decoder_hidden_dim)

        self.fc_out = nn.Linear((args.encoder_hidden_dim * 2) + args.decoder_hidden_dim + self.args.encoder_input_dim,
                                self.args.decoder_output_dim)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        input_seq = input_seq.view(1, 1, -1)

        # input = [1, batch size]

        # embedded = self.dropout(self.embedding(input_seq))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((input_seq, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        input_seq = input_seq.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, input_seq), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=self.device).float()


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, normalizer, args, device):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.normalizer = normalizer

    def forward(self, src, trg, teach_forcing_ratio):
        # src = [src len, batch size]
        # trg = [trg len, batch size]

        batch_size = self.args.batch_size
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        # outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        outputs = torch.zeros(self.args.max_pred_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        # torch.Size([3])
        first_input = trg[0, :]
        actual_seq_en = self.args.max_pred_len
        hit_finaL_subgoal = False

        for t in range(1, self.args.max_pred_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(first_input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teach_forcing_ratio

            # get the highest predicted token from our predictions
            # top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if teacher_force is True and t < trg_len:
                first_input = trg[t]
            else:
                first_input = output
            # print(trg[-1, :].shape)
            # print(output.shape)
            denormalized_output = self.normalizer.inverse_transform(output.clone().detach().cpu().numpy())
            denormalized_final_subgoal = self.normalizer.inverse_transform(trg[-1, :].clone().detach().cpu().numpy().reshape(1, -1))
            dist = numpy.linalg.norm(denormalized_output-denormalized_final_subgoal)
            if dist < self.args.final_goal_range and hit_finaL_subgoal is False:
                actual_seq_en = t
                hit_finaL_subgoal = True

        outputs = outputs[:trg_len]
        seq_len_diff = np.abs(actual_seq_en-trg_len)

        # first_input = trg[t] if teacher_force else output

        return outputs, seq_len_diff
