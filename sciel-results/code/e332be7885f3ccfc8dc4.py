"""Improved program using RBFs and scaled inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 4 Linear terms + 1 Quadratic term + 5 RBFs.
    Inputs scaled to [-0.75, 0.75] for conditioning.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Scale inputs: center at 1.25, range becomes [-0.75, 0.75]
    u = x - 1.25
    v = y - 1.25
    w = z - 1.25

    n = u.size
    features = np.empty((n, 10), dtype=np.float64)

    # 0: Constant
    features[:, 0] = 1.0
    # 1-3: Linear terms
    features[:, 1] = u
    features[:, 2] = v
    features[:, 3] = w
    
    # 4: Quadratic term (sum of squares)
    features[:, 4] = u*u + v*v + w*w

    # 5-9: RBFs (Gaussian kernels)
    # Sigma squared = 0.25 (Sigma = 0.5)
    # Centers chosen to cover the cube [-0.75, 0.75]^3
    sigma_sq = 0.25
    
    # Center 0: (0.5, 0.5, 0.5)
    r2 = (u - 0.5)**2 + (v - 0.5)**2 + (w - 0.5)**2
    features[:, 5] = np.exp(-r2 / sigma_sq)
    
    # Center 1: (-0.5, -0.5, -0.5)
    r2 = (u + 0.5)**2 + (v + 0.5)**2 + (w + 0.5)**2
    features[:, 6] = np.exp(-r2 / sigma_sq)
    
    # Center 2: (-0.5, 0.5, 0.5)
    r2 = (u + 0.5)**2 + (v - 0.5)**2 + (w - 0.5)**2
    features[:, 7] = np.exp(-r2 / sigma_sq)
    
    # Center 3: (0.5, -0.5, -0.5)
    r2 = (u - 0.5)**2 + (v + 0.5)**2 + (w + 0.5)**2
    features[:, 8] = np.exp(-r2 / sigma_sq)
    
    # Center 4: (0, 0, 0)
    r2 = u*u + v*v + w*w
    features[:, 9] = np.exp(-r2 / sigma_sq)

    return features @ p
# EVOLVE-BLOCK-END
