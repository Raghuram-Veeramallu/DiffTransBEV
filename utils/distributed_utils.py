import random

import torch
import numpy as np

def set_seed(seed: int = 0) -> None:
    """
    Set random seed for PyTorch, Python random, NumPy
    Sets CUDA convolution algorithm to be deterministic 
    Ref: https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
