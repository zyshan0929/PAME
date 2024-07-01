import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.activation import MultiheadAttention
from util.misc import positionalencoding1d


class CrossAttention(nn.Module):
    def __init__(self, dim_model=1024, nhead=8, dim_feedforward=1024, dropout=0.1) -> None:
        super(CrossAttention, self).__init__()
        self.attention = MultiheadAttention(dim_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2= nn.Dropout(dropout)

        self.activation = F.relu
        self.dim_model = dim_model

    def forward(self, guide:Tensor, src:Tensor) -> Tensor:
        """
        guide: [B, 1, dim_model]
        src:   [B, 6, dim_model]
        """
        
        
        # src2 = self.attention(guide, src, src)[0]
        # src = src + self.dropout(src2)
        # src  = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        # guide += positionalencoding1d(self.dim_model,1).cuda()
        # src += positionalencoding1d(self.dim_model,6).cuda()
        src2 = self.attention(guide, src, src)[0].squeeze(1)
        # src2 = self.dropout(src2)
        # src2 = self.norm1(torch.mean(src,dim=1) + src2)
        # output = self.norm2(src2 + self.activation(self.linear1(src2)))
        return src2

class guide_fusion(nn.Module):
    def __init__(self, embed_dim=768) -> None:
        super().__init__()
        self.cross_attn = CrossAttention(dim_model=embed_dim)
        self.query = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)
            # nn.BatchNorm1d(embed_dim)
        )
        self.kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)
            # nn.BatchNorm1d(embed_dim)
        )

    def forward(self, guide, imgs):
        """
        guide: [B, embed_dim]
        imgs:  [B, view_length, embed_dim]
        """
        query = self.query(guide).unsqueeze(1)
        B, view_length, _ = imgs.shape
        # imgs = imgs.view(B*view_length, -1)
        # kv = self.kv(imgs).view(B, view_length, -1)
        kv = self.kv(imgs)
        feat = self.cross_attn(query, kv).squeeze(1)
       
        return feat