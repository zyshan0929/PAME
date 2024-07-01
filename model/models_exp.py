from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
import model.models_mae as models_mae
from model.fusion import guide_fusion

class PAME(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.model_ref = models_mae.__dict__['mae_vit_base_patch16']()
        self.model_dist = models_mae.__dict__['mae_vit_base_patch16']()
        
    
    def forward(self, imgs, mask_ratio_dist=None, ref_imgs=None, mask_ratio_ref=None, mode='pretrain'):
        if mode == 'pretrain':
            assert mask_ratio_dist is not None and mask_ratio_ref is not None
            latent_ref, mask_ref, ids_restore_ref = self.model_ref.forward_encoder(imgs, mask_ratio_ref, mode)
            pred_ref = self.model_ref.forward_decoder(latent_ref, ids_restore_ref)
            # loss_ref = self.model_ref.forward_loss(imgs, pred_ref, mask_ref)
            target_ref = self.model_ref.patchify(ref_imgs)
            if self.model_ref.norm_pix_loss:
                mean_ref = target_ref.mean(dim=-1, keepdim=True)
                var_ref = target_ref.var(dim=-1, keepdim=True)
                target_ref = (target_ref - mean_ref) / (var_ref + 1.e-6)**.5
            loss_ref = (pred_ref - target_ref) ** 2
            loss_ref = loss_ref.mean(dim=-1)
            loss_ref = (loss_ref*mask_ref).sum() / mask_ref.sum()

            latent_dist, mask_dist, ids_restore_dist = self.model_dist.forward_encoder(imgs, mask_ratio_dist, mode)
            pred_dist = self.model_dist.forward_decoder(latent_dist, ids_restore_dist)
            loss_dist = self.model_dist.forward_loss(imgs, pred_dist, mask_dist)
            return loss_ref, loss_dist
        else:
            # mode=='finetune' to disable random masking in forward_encoder
            latent_ref = self.model_ref.forward_encoder(imgs, mask_ratio_ref, mode)
            latent_dist = self.model_dist.forward_encoder(imgs, mask_ratio_dist, mode)

            return latent_ref, latent_dist

class Fusion(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, dropout_rate=0.1) -> None:
        super().__init__()
        self.fusion = guide_fusion(embed_dim=embed_dim)
        self.regression = nn.Sequential(
            # nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats_ref, feats_dist):
        B, view_length, C = feats_ref.shape
        guide = feats_ref.max(dim=1)[0]
        feat = self.fusion(guide, feats_dist)
        pred_mos = self.regression(feat)
        return pred_mos


