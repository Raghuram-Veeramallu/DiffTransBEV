import torch.nn as nn

class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key_value):

        # Flatten spatial dimensions for query and key/value
        query_flat = query.flatten(2).permute(2, 0, 1)  # Shape: [Spatial, Batch, Channels]
        kv_flat = key_value.flatten(2).permute(2, 0, 1)  # Shape: [Spatial, Batch, Channels]

        # Apply cross-attention
        attn_output, _ = self.multihead_attn(query=query_flat, key=kv_flat, value=kv_flat)

        # Reshape output to original dimensions
        attn_output_reshaped = attn_output.permute(1, 2, 0).view_as(query)

        return attn_output_reshaped
