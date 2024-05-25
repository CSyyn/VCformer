import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# B：batch size
# M：多变量序列的变量数
# L：过去序列的长度
# T: 预测序列的长度
# N: 分Patch后Patch的个数
# D：每个变量的通道数
# P：kernel size of embedding layer
# S：stride of embedding layer


class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=2048):
        super(Embedding, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=D,
            kernel_size=P,
            stride=S
        )

    def forward(self, x):
        # x: [B, M, L]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
        x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
        x_pad = F.pad(
            x,
            pad=(0, self.P-self.S),
            mode='replicate'
        )  # [B*M, 1, L] -> [B*M, 1, L+P-S]

        x_emb = self.conv(x_pad)  # [B*M, 1, L+P-S] -> [B*M, D, N]
        # [B*M, D, N] -> [B, M, D, N]
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)

        return x_emb  # x_emb: [B, M, D, N]


class ConvFFN(nn.Module):
    def __init__(self, M, D, r, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        groups_num = M if one else D
        self.pw_con1 = nn.Conv1d(
            in_channels=M*D,
            out_channels=r*M*D,
            kernel_size=1,
            groups=groups_num
        )
        self.pw_con2 = nn.Conv1d(
            in_channels=r*M*D,
            out_channels=M*D,
            kernel_size=1,
            groups=groups_num
        )

    def forward(self, x):
        # x: [B, M*D, N]
        x = self.pw_con2(F.gelu(self.pw_con1(x)))
        return x  # x: [B, M*D, N]


class ModernTCNBlock(nn.Module):
    def __init__(self, M, D, kernel_size, r):
        super(ModernTCNBlock, self).__init__()
        # 深度分离卷积负责捕获时域关系

        self.dw_conv = nn.Conv1d(
            in_channels=M*D,
            out_channels=M*D,
            kernel_size=kernel_size,
            groups=M*D,
            padding='same'
        )
        self.bn = nn.BatchNorm1d(M*D)
        self.conv_ffn1 = ConvFFN(M, D, r, one=True)
        self.conv_ffn2 = ConvFFN(M, D, r, one=False)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        D = x_emb.shape[-2]
        # [B, M, D, N] -> [B, M*D, N]
        x = rearrange(x_emb, 'b m d n -> b (m d) n')
        # [B, M*D, N] -> [B, M*D, N]
        x = self.dw_conv(x)
        # [B, M*D, N] -> [B, M*D, N]
        x = self.bn(x)
        # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_ffn1(x)

        # [B, M*D, N] -> [B, M, D, N]
        x = rearrange(x, 'b (m d) n -> b m d n', d=D)
        # [B, M, D, N] -> [B, D, M, N]
        x = x.permute(0, 2, 1, 3)
        # [B, D, M, N] -> [B, D*M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')

        # [B, D*M, N] -> [B, D*M, N]
        x = self.conv_ffn2(x)

        # [B, D*M, N] -> [B, D, M, N]
        x = rearrange(x, 'b (d m) n -> b d m n', d=D)
        # [B, D, M, N] -> [B, M, D, N]
        x = x.permute(0, 2, 1, 3)

        out = x + x_emb

        return out  # out: [B, M, D, N]


class Model(nn.Module):
    def __init__(self, configs, P=8, S=4, kernel_size=51):
        super(Model, self).__init__()
        self.M = configs.enc_in
        self.L = configs.seq_len
        self.T = configs.pred_len
        self.D = configs.d_model
        self.r = configs.d_ff//configs.d_model
        self.num_layers = configs.e_layers
        # 深度分离卷积负责捕获时域关系
        self.N = self.L // S
        self.embed_layer = Embedding(P, S, configs.d_model)
        self.backbone = nn.ModuleList(
            [ModernTCNBlock(configs.enc_in, configs.d_model, kernel_size, self.r) for _ in range(self.num_layers)])
        self.head = nn.Linear(self.D*self.N, self.T)

    def forward(self, x, x_mark, x_dec, x_dec_mark):
        # x: [B, L, M]
        x = rearrange(x, 'b l m -> b m l')
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]

        # Flatten
        # [B, M, D, N] -> [B, M, D*N]
        z = rearrange(x_emb, 'b m d n -> b m (d n)')
        pred = self.head(z)  # [B, M, D*N] -> [B, M, T]
        pred = rearrange(pred, 'b m t -> b t m ')
        return pred  # out: [B, T, M]


# past_series = torch.rand(2, 4, 96)
# model = ModernTCN(4, 96, 192)
# pred_series = model(past_series)
# print(pred_series.shape)
# # torch.Size([2, 4, 192])
