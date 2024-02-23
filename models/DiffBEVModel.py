import torch
import torch.nn as nn
from models.attention.cross_attention import CrossAttention
from models.diffusion.dpm import DPM
from models.encode_head.bevencode import BevEncode

from models.feature_extractor.swin_transformer import SwinV2Transformer
from models.img_backbone.unet import UNet
from models.view_transformer.lss import LiftSplatShootTransformer
from models.view_transformer.lss_utils import get_grid_config

class DiffBEVModel(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # initializing the SWin Transformer module
        swin_pretrained = self.config.model.swin_pretrained
        self.swin_transformer = SwinV2Transformer(self.config, swin_pretrained)

        # initializing the Lift Splat Shoot module
        lss_grid_config = get_grid_config(self.config)
        self.lss_transformer = LiftSplatShootTransformer(lss_grid_config, device=self.device)

        # initializing the UNet module
        unet_input_channels = self.config.unet.input_channels
        self.ddpm_unet = UNet(input_channels=unet_input_channels)

        # initializing the Diffusion module
        beta_start = self.config.diffusion.beta_start
        beta_end = self.config.diffusion.beta_end
        num_timesteps = self.config.diffusion.num_timesteps
        self.diffusion_pipeline = DPM(beta_start, beta_end, num_timesteps, device=self.device)

        # initalizing the cross attention module
        embed_dim = self.config.crossattention.embed_dim
        num_heads = self.config.crossattention.num_heads
        self.cross_attention = CrossAttention(embed_dim, num_heads)

        # initializing the Bev encoding module
        inC = self.config.bevencode.input_channels
        outC = self.config.bevencode.output_channels
        self.lss_bev_encode = BevEncode(inC, outC)


    def forward(self, x, rots, trans, intrins, anns):
        # input shape: [6 x 3 x 225 x 400]

        # Forward pass through the Swin Transformer
        # output shape: [6 x 768 x 8 x 13]
        feature_maps = self.swin_transformer.forward(x)

        # Reshape to [B, N, C, ImH, ImW] for LSS input
        N_prime, C, ImH, ImW = feature_maps.shape
        # computing actual batch size (6 images for one instance)
        N = int(N_prime / 6)
        # just adding one batch for feature maps for LSS
        feature_maps = feature_maps.view(N, 6, C, ImH, ImW)

        # feature_maps shape: [1 x 6 x 768 x 8 x 13]

        # conditional_signal shape: ([1, 64, 100, 100])
        lss_out, _ = self.lss_transformer.forward(feature_maps, rots, trans, intrins)

        # Sample a random timestep for each image (lss_out.shape[0] is the batch size)
        timesteps = torch.randint(0, self.diffusion_pipeline.num_timesteps, (lss_out.shape[0],)).long()
        timesteps = timesteps.to(self.device)

        noisy_images, noise = self.diffusion_pipeline.forward_diffusion(lss_out, timesteps)
        noisy_images = noisy_images.to(self.device)

        # Predict the noise residual
        noise_pred = self.ddpm_unet(noisy_images, timesteps)

        # merge noise with the actual output
        bev_repr = self.cross_attention(lss_out, noise_pred)

        out = self.lss_bev_encode(bev_repr)
        return out
