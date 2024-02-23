import numpy as np

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from models.diffusion.dit_components.final_layer import FinalLayer

import models.diffusion.dit_components.gaussian_diffusion as gd
from models.diffusion.dit_components.dit_adalnz import DiTBlockAdaLNZ
from models.diffusion.dit_components.label_embedder import LabelEmbedder
from models.diffusion.dit_components.timestep_embedder import TimestepEmbedder
from models.diffusion.resample_utils import create_named_schedule_sampler
from models.diffusion.respace_utils import SpacedDiffusion, space_timesteps

class DPMDiT(nn.Module):
    # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py

    def __init__(self, 
                 image_size=100,
                 patch_size=4,
                 in_channels=64,
                 hidden_size=256,
                 num_classes=1000,
                 class_dropout_prob=0.1,
                 num_heads=16,
                 depth = 14,
                 mlp_ratio = 4.0,
                 channel_mult="",
                 attention_resolutions="16,8",
                 schedule_sampler="uniform"):
        super().__init__()

        # if channel_mult == "":
        #     if image_size == 512:
        #         channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        #     elif image_size == 256:
        #         channel_mult = (1, 1, 2, 2, 4, 4)
        #     elif image_size == 128:
        #         channel_mult = (1, 1, 2, 3, 4)
        #     elif image_size == 64:
        #         channel_mult = (1, 2, 3, 4)
        #     else:
        #         raise ValueError(f"Unsupported image size: {image_size}")
        # else:
        #     channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

        channel_mult = (1, 2, 3, 4)

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = in_channels

        ##################### For DiT #####################
        self.model = DiTBlockAdaLNZ(
            hidden_size=128,
            num_heads = 4, 
            mlp_ratio=4.0
        )

        self.x_embedder = PatchEmbed(image_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlockAdaLNZ(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        ##################### For DiT #####################

        # defining the diffusion model
        self.diffusion = self.create_gaussian_diffusion(
            steps=1000,
            learn_sigma=False,
            noise_schedule="linear",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            timestep_respacing="",
        )

        schedule_sampler = create_named_schedule_sampler(schedule_sampler, self.diffusion)

        # figure out how to run the training process

    def create_gaussian_diffusion(self,
                                  steps=1000, noise_schedule="linear", 
                                  use_kl = False, rescale_learned_sigmas = False,
                                  sigma_small = False, rescale_timesteps = False,
                                  predict_xstart = False, learn_sigma=False,
                                  timestep_respacing=""):

        betas = gd.get_named_beta_schedule(noise_schedule, steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if not timestep_respacing:
            timestep_respacing = [steps]

        return SpacedDiffusion(
            use_timesteps = space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = self.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0.1)
        nn.init.constant_(self.final_layer.linear.bias, 0.1)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False, extra_tokens=0):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def forward(self, x, t, y=None):
        # torch.Size([1, 625, 256])
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # torch.Size([1, 256])
        # unlike DiT we are not using labels as part of the conditional embedding
        c = self.t_embedder(t)                   # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
