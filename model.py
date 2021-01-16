import torch.nn as nn
from gat_block import GATBlock
from encoder_decoder import EncoderDecoder


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, node_number):
        super(Model, self).__init__()

        self.gat_block = nn.ModuleList((
            nn.BatchNorm2d(input_dim),
            GATBlock(input_dim, hidden_dim, stride=1, residual=True),
        ))

        self.EncoderDecoder = EncoderDecoder(input_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             output_dim=output_dim,
                                             node_number=node_number)

    def forward(self, agent_feature, adjacency, steps):
        feature = agent_feature
        for gat in self.gat_block:
            if type(gat) is nn.BatchNorm2d:
                feature = gat(feature)
            else:
                feature = gat(feature, adjacency)

        graph_feature = feature.permute(0, 2, 1, 3).unsqueeze(2).contiguous()
        last_location = agent_feature.unsqueeze(1).permute(0, 1, 3, 4, 2).contiguous()[:, :, -1:]

        predicted_xy = self.EncoderDecoder(in_data=graph_feature,
                                           last_location=last_location,
                                           steps=steps)
        predicted_xy = predicted_xy.permute(0, 2, 1, 3).contiguous()
        return predicted_xy
