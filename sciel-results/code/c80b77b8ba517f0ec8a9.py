"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Chebyshev polynomials (scaled to O(1)) + 5 Gaussian RBFs.
    Chebyshev provides global smooth approximation; RBFs add local flexibility.
    All basis functions are scaled to have magnitude O(1) for good BFGS conditioning.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0

    # Chebyshev Polynomials T_0 to T_4 (inherently bounded in [-1,1])
    f0 = np.ones_like(t)
    f1 = t
    f2 = 2.0 * t**2 - 1.0
    f3 = 4.0 * t**3 - 3.0 * t
    f4 = 8.0 * t**4 - 8.0 * t**2 + 1.0

    # Gaussian RBFs centered at -3, -1.5, 0, 1.5, 3 with sigma=1.0
    # Scale by 1/sqrt(2*pi*sigma^2) to keep magnitude reasonable
    sigma = 1.0
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    rbf_scale = 1.0 / np.sqrt(2.0 * np.pi * sigma**2)
    
    f5 = rbf_scale * np.exp(-0.5 * ((x - centers[0]) / sigma)**2)
    f6 = rbf_scale * np.exp(-0.5 * ((x - centers[1]) / sigma)**2)
    f7 = rbf_scale * np.exp(-0.5 * ((x - centers[2]) / sigma)**2)
    f8 = rbf_scale * np.exp(-0.5 * ((x - centers[3]) / sigma)**2)
    f9 = rbf_scale * np.exp(-0.5 * ((x - centers[4]) / sigma)**2)

    # Build feature matrix (N, 10)
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))

    # Linear combination
    result = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result

# EVOLVE-BLOCK-END
