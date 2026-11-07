# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-hidden-layer tanh network where the 10 params are the final output-layer
    weights. The hidden layers have fixed, diverse random projections so the model
    is linear in `params` (good for BFGS) but can represent rich non-linear functions.
    
    Architecture:
      z1 = tanh(W1 @ [x, y, 1]^T + b1)   # W1 fixed, shape (h1, 3)
      z2 = tanh(W2 @ z1 + b2)            # W2 fixed, shape (h2, h1+1)
      out = params @ z2                  # params shape (10,), z2 shape (n, 10)
    
    We use h1=8, h2=10 so that z2 has 10 features and params has length 10.
    """
    n = len(x)
    # Build input matrix: [x, y, 1] shape (n, 3)
    X = np.column_stack([x, y, np.ones(n)])
    
    # Fixed random weights for hidden layers (deterministic seed for reproducibility)
    rng = np.random.RandomState(42)
    h1 = 8
    h2 = 10
    W1 = rng.randn(h1, 3) * np.sqrt(2.0 / 3)
    b1 = rng.randn(h1) * 0.1
    W2 = rng.randn(h2, h1 + 1) * np.sqrt(2.0 / (h1 + 1))
    b2 = rng.randn(h2) * 0.1
    
    # Forward pass
    # z1: (n, h1)
    z1 = np.tanh(X @ W1.T + b1)
    # Append bias for next layer
    z1_aug = np.column_stack([z1, np.ones(n)])  # (n, h1+1)
    # z2: (n, h2)
    z2 = np.tanh(z1_aug @ W2.T + b2)
    
    # Output: linear combination of z2 with params
    # params shape (10,), z2 shape (n, 10)
    return z2 @ params
# EVOLVE-BLOCK-END
