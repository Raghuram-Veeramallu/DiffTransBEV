import os

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

from mmdet.models.backbones import SwinTransformer

class SwinV2Transformer(nn.Module):
    
    def __init__(self, config, pretrained=True):
        super().__init__()
        
        if pretrained:
            if not os.path.exists('./trained_models/swin_t_imagenet1k_v1.pth'):
                # Load pretrained Swin Transformer
                swin_pretrained_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
                torch.save(swin_pretrained_model.state_dict(), './trained_models/swin_t_imagenet1k_v1.pth')
            # Specify init_cfg
            init_cfg = dict(type='Pretrained', checkpoint='./trained_models/swin_t_imagenet1k_v1.pth')
        else:
            init_cfg = None

        self.model = SwinTransformer(
            pretrain_img_size=224,
            # patch partitioning
            patch_size=4,
            # linear embedding
            embed_dims=96,
            strides=(4, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            depths=(2, 2, 6, 2),
            norm_cfg=dict(type='LN', requires_grad=True),
            # Keep this int. float is a deprecated version and will cause errors
            mlp_ratio=4,
            # for shifting window
            out_indices=(3,),
            window_size=7,
            drop_rate=0,
            attn_drop_rate=0.,
            drop_path_rate=0.0,
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            init_cfg=init_cfg,
        )

        # if pretrained:
        #     weights = Swin_T_Weights.IMAGENET1K_V1
        #     self.model = swin_t(weights=weights)
        # else:
        #     self.model = swin_t(weights=None)

    def forward(self, x):
        # input: [B x C x H x W]
        feature_maps = self.model.forward(x)
        # output: [B x C' x ImH x ImW]
        # here C' is (C * 256)
        return feature_maps[-1]
