import torch.nn as nn
import torch
from einops import rearrange
#double sampling block由两部分组成,一部分是downsampling,一部分是piecewise sampling
#使用downsampling得到序列的多个子序列之后,将多个子序列拼接起来
#使用piecewise sampling类似于patch的过程,将多个patch拼接起来
#两种sampling都可以用einops中的rearange实现

    


class DoubleSamplingBlock(nn.Module):
    def __init__(self,configs,C) -> None:
        #C:int,downsampling interval or piecewise sampling interval
        super().__init__()
        self.C = C
        #downsampling
    def forward(self,x):
        #直接通过两次rearrange来实现downsampling&concat的操作
        x_ds=rearrange(x,"b c (l1 l2) -> b c l1 l2",l2=self.C)  
        x_ds=rearrange(x_ds,"b c l1 l2 -> b c l2 l1")
        #piecewise sampling
        x_ps=rearrange(x,"b c (l1 l2) -> b c l1 l2",l1=self.C)
        return x_ds,x_ps        
        