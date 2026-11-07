"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process-like kernel regression with 10 fixed centers.
    params: 10 weights for the kernel contributions.
    
    We use squared-exponential kernels:
    K_i(x) = exp(-gamma * (x - c_i)^2)
    
    where c_i are evenly spaced centers in [0.1, 6.0] and gamma is fixed.
    To allow the model to capture the mean level, we include a constant basis.
    Actually, with 10 params, let's use 9 kernel weights and 1 global offset?
    Or just 10 kernel weights. Since the domain is [0.1, 6], the sum of kernels
    won't necessarily sum to 1, so a linear combination of them is flexible.
    
    Let's define 10 centers evenly spaced in [0.1, 6.0].
    c_i = 0.1 + i * (5.9 / 9) for i in 0..9.
    
    To ensure numerical stability and good conditioning, we can normalize the kernels
    or just let BFGS handle it. The squared exponential kernel is smooth and local.
    This should approximate any smooth function well if the centers are dense enough.
    
    Let's try this approach.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed gamma (inverse length scale squared). 
    # A moderate gamma allows for local adaptation.
    # Let's try gamma = 2.0 or 5.0. 
    # If gamma is too small, it's like a global basis. If too large, it's very local.
    # With 10 points, a local model might be good.
    gamma = 5.0
    
    # Compute kernels
    # diff = x[:, None] - centers[None, :]
    # K = np.exp(-gamma * diff**2)
    # pred = K @ params
    
    # Vectorized computation
    diff = x[:, None] - centers[None, :]
    K = np.exp(-gamma * diff ** 2)
    
    pred = K @ params
    
    return pred
# EVOLVE-BLOCK-END
