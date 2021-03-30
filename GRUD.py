# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:48:54 2018

@author: Zhiyong
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


class DiagLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(DiagLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
        self.diag_matrix = torch.eye(in_features, requires_grad=False).to(device)      
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        #self.reset_parameters()
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.diag_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", bias="
            + str(self.bias is not None)
            + ")"
        )


class GRUD(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_dense=0.3, dropout_gru=0.3):
        """
        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        cell_size is the size of cell_state.

        Implemented based on the paper:
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }

        GRU-D:
            input_dim: variable dimension of each time
            hidden_dim: dimension of hidden_state
        """

        super(GRUD, self).__init__()

        self.hidden_dim = hidden_dim
        self.interval_dim = input_dim
        self.mask_dim = input_dim
        self.input_dim = input_dim
        self.dropout_dense = dropout_dense
        if self.dropout_dense:
            self.nn_drop_dense = nn.Dropout(p=self.dropout_dense)
        self.dropout_gru = dropout_gru
        if self.dropout_gru:
            self.nn_drop_gru = nn.Dropout(p=self.dropout_gru)
   
        gate_dim = self.mask_dim + self.hidden_dim + self.interval_dim
        self.zl = nn.Linear(gate_dim, hidden_dim)
        self.rl = nn.Linear(gate_dim, hidden_dim)
        self.hl = nn.Linear(gate_dim, hidden_dim)
        nn.init.xavier_uniform_(self.zl.weight)
        nn.init.zeros_(self.zl.bias)
        nn.init.orthogonal_(self.rl.weight)
        nn.init.zeros_(self.rl.bias)
        #self.embedding = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.gamma_x_l = nn.Linear(self.interval_dim, self.interval_dim)
        self.gamma_h_l = nn.Linear(self.interval_dim, self.hidden_dim)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)       
        self.nn_output = nn.Linear(self.hidden_dim, 1)

    def step(self, x, x_last, x_mean, mask, interval, h):
        gamma_x = torch.exp(-F.relu(self.gamma_x_l(interval)))
        gamma_h = torch.exp(-F.relu(self.gamma_h_l(interval)))
        x = mask * x + (1 - mask) * (gamma_x * x_last + (1 - gamma_x) * x_mean)
        #x = self.embedding(x)
        h = gamma_h * h
        combined = torch.cat((x, h, mask), 1)
        
        if self.dropout_gru:
            combined = self.nn_drop_gru(combined)
        z = torch.sigmoid(self.zl(combined))
        r = torch.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        if self.dropout_gru:
            combined_r = self.nn_drop_gru(combined_r)
        h_tilde = torch.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde
        
        return h

    def forward(self, x, x_last, interval, mask, device):
        batch_size = x.size(0)
        time = x.size(1)
        x = x.unsqueeze(dim=-1)
        x_last = x_last.unsqueeze(dim=-1)
        interval = interval.unsqueeze(dim=-1)
        mask = mask.unsqueeze(dim=-1)
        # be careful with calculating the mean, not to include the values with mask=0
        x_mean = torch.mean(x * mask, dim=1) * x.size(1) / mask.sum(axis=1)
        hidden_state = torch.zeros(batch_size, self.hidden_dim).to(device)
        output = None
        for t in range(time):
            hidden_state = self.step(
                x[:, t, :],
                x_last[:, t, :],
                x_mean,
                mask[:, t, :],
                interval[:, t, :],
                hidden_state
            )
        output = hidden_state
        if self.dropout_dense:
            output = self.nn_drop_dense(output)
        #output = self.batch_norm(output)
        output = self.nn_output(output)
        return torch.sigmoid(output)