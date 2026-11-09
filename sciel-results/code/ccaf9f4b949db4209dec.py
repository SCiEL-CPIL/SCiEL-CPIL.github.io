import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Linear trend + 8 Gaussian RBFs.
    Input x is scaled to [-1, 1] for numerical stability.
    RBF centers are evenly spaced in the scaled domain.
    """
    N = len(x)
    if N == 0:
        return np.array([])
        
    # Scale x to [-1, 1]
    t = x / 3.0
    
    # Parameters:
    # p0: Intercept
    # p1: Linear coefficient (in scaled space)
    # p2..p9: RBF amplitudes
    
    # Linear part
    y = params[0] + params[1] * t
    
    # RBF part
    # 8 centers evenly spaced in [-1, 1]
    centers = np.linspace(-1.0, 1.0, 8)
    
    # Width in scaled space
    # Original width ~1.0, so scaled width ~1/3
    # Using 0.35 for slightly wider support to ensure coverage and smoothness
    sigma = 0.35
    
    # Compute RBF values
    # t shape: (N,)
    # centers shape: (8,)
    # diff shape: (N, 8)
    diff = t[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Combine with amplitudes
    y += rbf_values @ params[2:]
    
    return y
