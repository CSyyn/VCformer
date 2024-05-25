import torch
from torch import nn, optim
import torch.nn.functional as F
from layers.Embed import DSEmbedding
from layers.DSformer_block import TVA_block_att
from layers.DSformer_Dec import TVADE_block
from layers.RevIn import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        """
        Input_len: History length
        out_len：future length
        num_id：number of variables
        num_layer：number of layer. 1 or 2
        muti_head：number of muti_head attention. 1 to 4
        dropout：dropout. 0.15 to 0.3
        num_samp：muti_head subsequence. 2 or 3
        IF_node:Whether to use node embedding. True or False
        """
        super(Model, self).__init__()
        Input_len = configs.seq_len
        out_len = configs.pred_len
        num_id = configs.enc_in
        num_layer = configs.e_layers
        dropout = configs.dropout
        muti_head = configs.n_heads
        num_samp = configs.sample_interval
        IF_node = configs.IF_node

        if IF_node:
            self.inputlen = 2 * Input_len // num_samp
        else:
            self.inputlen = Input_len // num_samp

        ### embed and encoder
        self.RevIN = RevIN(num_id)
        self.embed_layer = DSEmbedding(Input_len, num_id, num_samp, IF_node)
        self.encoder = TVA_block_att(
            self.inputlen, num_id, num_layer, dropout, muti_head, num_samp)
        self.laynorm = nn.LayerNorm([self.inputlen])

        # decorder
        self.decoder = TVADE_block(self.inputlen, num_id, dropout, muti_head)
        self.output = nn.Conv1d(in_channels=self.inputlen,
                                out_channels=out_len, kernel_size=1)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # Input [B,H,N]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N]: B is batch size. N is the number of variables. L is the future length

        # embed
        x = self.RevIN(x, 'norm').transpose(-2, -1)
        x_1, x_2 = self.embed_layer(x)

        # encoder
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        x = x_1 + x_2
        x = self.laynorm(x)

        # decorder
        x = self.decoder(x)
        x = self.output(x.transpose(-2, -1))
        x = self.RevIN(x, 'denorm')
        return x
