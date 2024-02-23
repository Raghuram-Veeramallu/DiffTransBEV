import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

from models.utils import modulate

class DiTBlockAdaLNZ(nn.Module):
    """
        A DiT block with adaptive layer (DiT-adaLN-Zero) normalization with zero initializing (adaLN-Zero) condition.

        Input
            - Input tokens (x)
            - Conditiong (c)
        
        2. First Stage:
            - Layer Normalization with Scale and Shift parameters (gamma1, beta1) from MLP (conditioned on Input Tokens)
            - Multi-Head Self-Attention mechanism
            - Scale operation controlled by alpha1
            - Residual connection adding back Input Tokens

        3. Second Stage:
            - Layer Normalization with Scale and Shift parameters (gamma2, beta2)
            - Pointwise Feedforward neural network
            - Scale operation controlled by alpha2
            - Residual connection from the output of First Stage

        Output: Resultant tokens after processing through the DiT block with adaLN-Zero.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # layer norm 1
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # multi-head self-attention
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        # layer norm 2
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # gaussian error activation layer
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        # MLP
        self.mlp = Mlp(
            in_features = hidden_size,
            hidden_features = int(hidden_size * mlp_ratio),
            act_layer = approx_gelu,
            drop = 0,
        )

        self.adaLN_modulation = nn.Sequential(
            # sigmoid linear unit
            nn.SiLU(),
            # linear
            # 6 * hidden_size for 6 parameters (2 x (gamma, beta, alpha))
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    # Here, x is the input tokens and c is the conditioning
    def forward(self, x, c):
        ### beta -> shift
        ### gamma -> scale
        ### alpha ->  dimension-wise scaling
        beta1, gamma1, alpha1, beta2, gamma2, alpha2 = self.adaLN_modulation(c).chunk(6, dim=1)
        
        ### block 1
        ## Input -> Layer Norm -> Scale and Shift with gamma1, beta1 -> Multi-Head Self-Attention -> Dimension-wise scaling with alpha1 + original i/p -> Output
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.norm1(x), beta1, gamma1))

        ### block 2
        ## Block 1 o/p -> Layer Norm -> Scale and Shift with gamma2, beta2 -> Pointwise Feedforward -> Dimension-wise scaling with alpha2 + block 1 o/p -> Output
        x = x + alpha2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), beta2, gamma2))
        return x
