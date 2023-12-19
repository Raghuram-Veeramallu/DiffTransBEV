####
# Image View Backbone: Extracts image features from the input
# Input: An image or image-like data (H x W x C)
# Output: Extracted image features
####

####
# View Transformer: Lifts the image-view features to a bird's-eye view (BEV)
#### USED : LSS
# Input: Extracted image features from the Image View Backbone
# Output: BEV features
####

####
# Conditional Diffusion Model (Conditional DPM): Refines noisy samples progressively to generate high-quality semantic features.
#### USED: U-Net Backbone
# Input: The features generated by the view transformer which are treated as the condition of the diffusion model. 
# It uses noise xT which obeys a standard normal distribution and transforms it to x0 in a progressive manner.
# Output: A refined, high-quality BEV semantic feature which is the outcome of a denoising process.
####
    
####
# Cross Attention Module: Refines the original BEV feature by leveraging the output of the conditional diffusion model and the original BEV feature.
# Input: Output of the Conditional Diffusion Model and the original BEV feature from the View Transformer.
# Output: A refined BEV feature.
####


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention.cross_attention import CrossAttention
from models.diffusion.dit import DPMDiT
from models.diffusion.dpm import DPM
from models.encode_head.bevencode import BevEncode
from models.feature_extractor.swin_transformer import SwinV2Transformer
from models.view_transformer.lss import LiftSplatShootTransformer
from models.view_transformer.lss_utils import get_grid_config

class DiffDiTBEV(nn.Module):

    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        # initializing the SWin Transformer module
        swin_pretrained = self.config.model.swin_pretrained
        self.swin_transformer = SwinV2Transformer(self.config, swin_pretrained)

        # initializing the Lift Splat Shoot module
        lss_grid_config = get_grid_config(self.config)
        self.lss_transformer = LiftSplatShootTransformer(lss_grid_config)

        # defining the number of diffusion steps
        self.num_diffusion_steps = self.config.dit.diffusion_steps

        # initializing the DiT module
        self.dpm_dit = DPMDiT()

        # initializing the Diffusion module
        beta_start = self.config.diffusion.beta_start
        beta_end = self.config.diffusion.beta_end
        num_timesteps = self.config.diffusion.num_timesteps
        self.diffusion_pipeline = DPM(beta_start, beta_end, num_timesteps)

        # initalizing the cross attention module
        embed_dim = self.config.crossattention.embed_dim
        num_heads = self.config.crossattention.num_heads
        self.cross_attention = CrossAttention(embed_dim, num_heads)

        # initializing the Bev encoding module
        inC = self.config.bevencode.input_channels
        outC = self.config.bevencode.output_channels
        self.lss_bev_encode = BevEncode(inC, outC)


    def forward(self, x, rots, trans, intrins, y):
        # N is the batch
        # input shape: [6B x 3 x 900 x 1600] (48 in case of batch size 8)

        # Forward pass through the Swin Transformer
        # output shape: [6B x 768 x 8 x 13]
        feature_maps = self.swin_transformer.forward(x)

        # Reshape to [B, N, C, ImH, ImW] for LSS input
        N_prime, _, _, _ = feature_maps.shape
        # computing actual batch size (6 images for one instance)
        N = int(N_prime / 6)
        if N > 1:
            chunks = torch.chunk(feature_maps, N, dim=0)
            feature_maps = torch.stack(chunks, dim=0)
        else:
            feature_maps = torch.unsqueeze(feature_maps, 0)

        # conditional_signal shape: ([B, 64, 100, 100])
        lss_out, _ = self.lss_transformer.forward(feature_maps, rots, trans, intrins)

        # Sample a random timestep for each image (lss_out.shape[0] is the batch size)
        # torch.Size([B])
        timesteps = torch.randint(0, self.diffusion_pipeline.num_timesteps, (lss_out.shape[0],),
                                    device=self.device).long()

        # shape: torch.Size([B, 64, 100, 100])
        noisy_images, noise = self.diffusion_pipeline.forward_diffusion(lss_out, timesteps)
        noisy_images = noisy_images.to(self.device)

        # Predict the noise residual
        # shape: torch.Size([B, 64, 100, 100])
        noise_pred = self.dpm_dit(noisy_images, timesteps, y)

        # merge noise with the actual output
        # torch.Size([B, 64, 100, 100])
        bev_repr = self.cross_attention(lss_out, noise_pred)

        # torch.Size([B, 1, 200, 200])
        out = self.lss_bev_encode(bev_repr)

        return out
 