import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.gcn import ConvGraphical
from net.utils.graph import Graph
import random

class Model(nn.Module):

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()
        random.seed(1)
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        self.register_buffer('A', A)
        # build networks
        skernel_size = A.size(0)
        tkernel_size = 1
        kernel_size = (tkernel_size, skernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.dgcn_networks = nn.ModuleList((
            dgcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            dgcn(64, 64, kernel_size, 1, **kwargs),
            dgcn(64, 64, kernel_size, 1, **kwargs),
            dgcn(64, 64, kernel_size, 1, **kwargs),
            dgcn(64, 128, kernel_size, 2, **kwargs),
            dgcn(128, 128, kernel_size, 1, **kwargs),
            dgcn(128, 128, kernel_size, 1, **kwargs),
            dgcn(128, 128, kernel_size, 1, **kwargs),
        ))

        n_class = 117

        hide_size = 64

        final_num = 128

        # fcn for prediction
        self.fcn = nn.Conv1d(final_num , final_num , kernel_size=1)

        self.tcn_f = nn.Conv1d(final_num , n_class, kernel_size=1)

        self.tcn_fn = nn.Conv1d(hide_size, 1, kernel_size=1)


        self.rnn = nn.RNN(input_size=A[0].shape[0], num_layers=4,
                          hidden_size=hide_size, batch_first=True, nonlinearity= 'tanh', bidirectional=False)

        self.linear = nn.Linear(final_num , n_class)

        self.Softmax = nn.Softmax(dim=1)

        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.A.size()))
            for i in self.dgcn_networks])

        self.tcn_f2 = nn.Sequential(
            nn.BatchNorm1d(n_class),

            nn.Conv1d(
                n_class,
                n_class,
                kernel_size=kernel_size[0],
                padding=0,
                stride=1,
            ),
            nn.BatchNorm1d(n_class),

        )

    def forward(self, x):

        n, v, c = x.size()

        x = x.permute(0, 2, 1).contiguous()
        x = x.view(n , v * c)
        x = self.data_bn(x)
        x = x.view(n, v, c)
        x = x.permute(0, 1, 2).contiguous()

        x = x.view(n, c, v)

        for gcn in self.dgcn_networks:
            x, _ = gcn(x, self.A)

        x2 = x.clone()

        x2 = self.tcn_f(x2)
        x2 = self.tcn_f2(x2)

        x2 = torch.exp(x2 - self.Softmax(x2))
        x2 = 1 / (1 + x2)


        self.rnn.flatten_parameters()

        x2, states = self.rnn(x2)

        x2 = x2.permute(0, 2, 1).contiguous()

        x2 = self.tcn_fn(x2)
        x2 = x2.squeeze()

        x = torch.avg_pool1d(x, x.size()[2:])

        x = self.fcn(x)
        x = x.squeeze()

        x = self.linear(x)

        x2 = torch.mean((x2 - self.Softmax(x2)) ** 2)

        c1 = 0.8
        c2 = 1 - c1

        x = (c1*x + c2*x2)/(c1 + c2)

        return x


    def extract_feature(self, x):
        N, C,  V = x.size()
        x = x.permute(0, 2, 1).contiguous()

        x = self.data_bn(x)
        x = x.view(N, V, C)
        x = x.permute(0, 1, 2).contiguous()
        x = x.view(N, C, V)

        _, c, v = x.size()
        feature = x.view(N, c, v).permute(0, 2, 1)

        x = self.fcn(x)
        output = x.view(N, -1, v).permute(0, 1, 2)
        return output, feature

class dgcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size[0],
                #  kernel_size=(kernel_size, kernel_size),
                padding=0,
                stride=1,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.dropout = nn.Dropout(0.25) # drop out layer with 20% dropped out neuron
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1),
                nn.BatchNorm1d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.leaky = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.dropout(x)
        x = self.tcn(x) + res
        x = F.relu(x)
        return x, A






