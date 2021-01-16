import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input, adj):
        h_prime_cat = torch.zeros(size=(input.shape[0],
                                        input.shape[2],
                                        input.shape[3],
                                        self.out_features)).to(input.device)

        for step_i in range(input.shape[2]):
            input_i = input[:, :, step_i, :]
            input_i = input_i.permute(0, 2, 1)
            adj_i = adj[:, step_i, :, :]
            Wh = torch.matmul(input_i, self.W)

            batch_size = Wh.size()[0]
            N = Wh.size()[1]  # number of nodes
            Wh_chunks = Wh.repeat(1, 1, N).view(batch_size, N * N, self.out_features)
            Wh_alternating = Wh.repeat(1, N, 1)
            combination_matrix = torch.cat([Wh_chunks, Wh_alternating], dim=2)
            a_input = combination_matrix.view(batch_size, N, N, 2 * self.out_features)

            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)

            attention = torch.where(adj_i > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, 0.25, training=self.training)
            h_prime = torch.matmul(attention, Wh)  # [8, 120, 64]
            h_prime_cat[:, step_i, :, :] = h_prime

        if self.concat:
            return F.elu(h_prime_cat)
            # return h_prime_return
        else:
            return h_prime_cat


class GATBlock(nn.Module):
    def __init__(self, input_dim, out_channels, stride=1, residual=True):
        super(GATBlock, self).__init__()

        self.att_1 = GraphAttentionLayer(input_dim, out_channels, concat=True)
        self.att_2 = GraphAttentionLayer(input_dim, out_channels, concat=True)
        self.att_out = GraphAttentionLayer(out_channels, out_channels, concat=False)

        if not residual:
            self.residual = lambda x: 0
        elif (input_dim == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels), )

    def forward(self, x, adjacency):
        res = self.residual(x)
        x_1 = self.att_1(x, adjacency)
        x_2 = self.att_2(x, adjacency)
        x = torch.stack([x_1, x_2], dim=-1).mean(-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.dropout(x, 0.25)
        x = F.elu(self.att_out(x, adjacency))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x + res
        return x
