
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        padding = kernel_size // 2
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input_tensor, hidden_state):
        # Ensure hidden state is on the same device as input_tensor
        if hidden_state is None:
            hidden_state = torch.zeros(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), input_tensor.size(3), device=input_tensor.device)
        combined = torch.cat([input_tensor, hidden_state], dim=1)  # Concatenate input and hidden state
        reset = self.sigmoid(self.reset_gate(combined))
        update = self.sigmoid(self.update_gate(combined))
        combined_r = torch.cat([input_tensor, reset * hidden_state], dim=1)
        out = self.tanh(self.out_gate(combined_r))
        new_hidden = (1 - update) * hidden_state + update * out
        return new_hidden

class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, num_layers):
        super(ConvGRU, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            self.layers.append(ConvGRUCell(in_size, hidden_sizes[i], kernel_sizes[i]))

    def forward(self, x, h=None):
        if h is None:
            h = [None] * self.num_layers
        for i, layer in enumerate(self.layers):
            x = layer(x, h[i])
            h[i] = x
        return x, h

class EncoderDecoder(nn.Module):
    def __init__(self, enc_input_size, enc_hidden_sizes, enc_kernel_sizes, enc_num_layers,
                 dec_input_size, dec_hidden_sizes, dec_kernel_sizes, dec_num_layers):
        super(EncoderDecoder, self).__init__()
        self.encoder = ConvGRU(enc_input_size, enc_hidden_sizes, enc_kernel_sizes, enc_num_layers)
        self.decoder = ConvGRU(dec_input_size, dec_hidden_sizes, dec_kernel_sizes, dec_num_layers)

    def forward(self, x_enc, x_dec, h_enc=None, h_dec=None):
        _, h_enc = self.encoder(x_enc, h_enc)
        out, h_dec = self.decoder(x_dec, h_dec=h_enc)  # Use encoder's final hidden state as decoder's initial state
        return out

# Example usage:
# model = EncoderDecoder(enc_input_size=3, enc_hidden_sizes=[32, 64], enc_kernel_sizes=[3, 3], enc_num_layers=2,
#                        dec_input_size=3, dec_hidden_sizes=[64, 32], dec_kernel_sizes=[3, 3], dec_num_layers=2)
# x_enc = torch.randn(5, 3, 224, 224)  # Example encoder input
# x_dec = torch.randn(5, 3, 224, 224)  # Example decoder input
# output = model(x_enc, x_dec)
