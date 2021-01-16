import torch
import torch.nn as nn
from layers.convgru import ConvGRU


class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, node_number):
        super(EncoderDecoder, self).__init__()
        self.encoder = ConvGRU(input_size=(hidden_dim, node_number), input_dim=1, hidden_dim=[45],
                               kernel_size=(5, 5), num_layers=1,
                               dtype=None, batch_first=True,
                               bias=True, return_all_layers=True)

        self.decoder_fc = nn.Linear(input_dim, hidden_dim)
        self.decoder = ConvGRU(input_size=(hidden_dim, node_number), input_dim=1, hidden_dim=[45],
                               kernel_size=(5, 5), num_layers=1,
                               dtype=None, batch_first=True,
                               bias=True, return_all_layers=True)
        self.dropout = nn.Dropout(0.5)
        self.decoder_out_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, in_data, last_location, steps):
        batch_size = in_data.shape[0]
        encoded_output, hidden = self.encoder(in_data)
        decoder_input = self.decoder_fc(last_location)
        decoder_input = decoder_input.permute(0, 1, 2, 4, 3).contiguous()

        outputs = torch.zeros(batch_size, steps, 64, 120).to(in_data.device)
        for step in range(steps):
            now_out, hidden = self.decoder(decoder_input, hidden[-1])
            now_out = self.dropout(now_out[0][:, :, -1:, :, :])
            now_out += decoder_input
            decoder_input = now_out
            outputs[:, step:step + 1] = now_out.squeeze(2)
        outputs = self.decoder_out_fc(outputs.permute(0, 1, 3, 2).contiguous())
        outputs = outputs.permute(0, 1, 3, 2).contiguous()
        return outputs
