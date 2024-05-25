import math
import torch
import torch.nn as nn
from einops import rearrange


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=2,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class KPLayerApprox(nn.Module):
    def __init__(self):
        super(KPLayerApprox, self).__init__()
        self.K = None
        self.K_step = None

    def forward(self, z):
        B, N, input_len, hidden_dim = z.shape
        pred_len = input_len
        z = rearrange(z, 'b n pn m -> (b n) pn m')
        x, y = z[:, :-1], z[:, 1:]

        self.K = torch.linalg.lstsq(x, y).solution

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(
                self.K.device).unsqueeze(0).repeat(B, 1, 1)

        self.K_step = torch.linalg.matrix_power(self.K, pred_len)
        if torch.isnan(self.K_step).any():
            print('Encounter multistep K with nan, replace it by identity matrix')
            self.K_step = torch.eye(self.K_step.shape[1]).to(
                self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
        z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        return z_pred


class KTDlayer(nn.Module):
    """
        Koopman Temporal Detector layer
    """

    def __init__(self, configs,
                 enc_in, snap_size, proj_dim, hidden_dim, hidden_layers):
        super(KTDlayer, self).__init__()
        self.enc_in = enc_in
        self.snap_size = snap_size
        self.dynamics = KPLayerApprox()
        self.encoder = MLP(f_in=snap_size, f_out=proj_dim,
                           hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.decoder = MLP(f_in=proj_dim, f_out=snap_size,
                           hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.padding_len = snap_size - \
            (enc_in % snap_size) if enc_in % snap_size != 0 else 0

    def forward(self, x):
        # x: B L D
        B, N, D = x.shape

        res = torch.cat((x[:, :, D-self.padding_len:], x), dim=-1)

        res = rearrange(res, 'b n (p_n p) -> b n p_n p', p=self.snap_size)

        res = self.encoder(res)  # b n p_n m, m means hidden dim

        # b*n f_n m, f_n means forecast patch num
        x_pred = self.dynamics(res)

        x_pred = self.decoder(x_pred)     # b*n f_n p

        x_pred = rearrange(x_pred, '(b n) f_n p -> b n (f_n p)', b=B)

        return x_pred
