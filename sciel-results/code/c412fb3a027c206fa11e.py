"""Symbolic regression with portfolio-of-forms strategy and time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Portfolio-of-forms strategy: try multiple model structures and select the best.
    
    Uses 10 parameters total across a hybrid model structure that combines
    rational, exponential, and logarithmic components for robustness.
    
    Structure:
    y = (p0 + p1*u + p2*u^2 + p3*u^3) / (1 + p4*u^2) + p5*exp(p6*u) + p7*log(x+1) + p8*x + p9
    where u = (x - 3.05) / 3.0 (standardized)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    x = np.asarray(x, dtype=np.float64)
    x_safe = np.clip(x, 1e-10, None)
    
    # Standardized input
    u = (x_safe - 3.05) / 3.0
    
    # Build a rich basis with 10 components
    b0 = np.ones_like(x_safe)
    b1 = u
    b2 = u ** 2
    b3 = u ** 3
    b4 = np.exp(np.clip(u, -50, 50))
    b5 = np.exp(-np.clip(u, -50, 50))
    b6 = np.log(x_safe + 1.0)
    b7 = 1.0 / (x_safe + 1.0)
    b8 = x_safe
    b9 = np.sin(u)
    
    # Stack into matrix: shape (n, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Predict: (n, 10) @ (10,) -> (n,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
