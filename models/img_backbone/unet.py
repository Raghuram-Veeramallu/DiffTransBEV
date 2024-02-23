import torch
import torch.nn as nn
import torch.nn.functional as F

from models.img_backbone.unet_layers import AttentionDownBlock, AttentionUpBlock, ConvDownBlock, ConvUpBlock, TransformerPositionalEmbedding

class UNet(nn.Module):

    def __init__(self, input_channels=3):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding='same',
        )

        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(in_channels=64, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvDownBlock(in_channels=128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvDownBlock(in_channels=128, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            AttentionDownBlock(in_channels=256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4),
            ConvDownBlock(in_channels=256, out_channels=512, num_layers=2, num_groups=32, time_emb_channels=128 * 4)
        ])

        self.bottleneck = AttentionDownBlock(in_channels=512, out_channels=512, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128*4, downsample=False)                                                                                                  # 16x16x256 -> 16x16x256

        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(in_channels=512 + 512, out_channels=512, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            AttentionUpBlock(in_channels=512 + 256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4),
            ConvUpBlock(in_channels=128 + 128, out_channels=64, num_layers=2, num_groups=32, time_emb_channels=128 * 4)
        ])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=128, num_groups=32),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        time_encoded = self.positional_encoding(t)
        # torch.Size([1, 64, 98, 100])
        initial_x = self.initial_conv(x)

        states_for_skip_connections = [initial_x]

        x = initial_x
        x_shapes = []
        x_shapes.append(x.shape)
        for i, block in enumerate(self.downsample_blocks):
            x = block(x, time_encoded)
            x_shapes.append(x.shape)
            states_for_skip_connections.append(x)
        states_for_skip_connections = list(reversed(states_for_skip_connections))
        x_shapes = list(reversed(x_shapes))

        x = self.bottleneck(x, time_encoded)

        for i, (block, skip, x_shape) in enumerate(zip(self.upsample_blocks, states_for_skip_connections, x_shapes)):
            # no issue over here
            # to keep the shapes sames after downsamping and upsampling
            if x.shape[2:] != x_shape[2:]:
                x = F.interpolate(x, size=x_shape[2:], mode='bilinear')
            # looks like attention down and attention up are causing trouble
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded)

        # Concat initial_conv with tensor
        x = torch.cat([x, states_for_skip_connections[-1]], dim=1)

        # Get initial shape [3, 256, 256] with convolutions
        out = self.output_conv(x)

        return out
