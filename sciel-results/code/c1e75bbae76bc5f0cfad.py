"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) network with shifted centers.
    
    params[0:5]  : offsets for the centers of the 5 Gaussian kernels
    params[5:10] : amplitudes of the 5 Gaussian kernels
    
    The centers are defined as params[0:5] + fixed_offsets.
    The fixed_offsets are chosen such that when params are initialized to 1,
    the centers are spread across the domain [-2, 2], helping BFGS escape
    local minima caused by symmetric initialization.
    """
    # Fixed offsets to spread initial centers (when params=1) across [-2, 2]
    # Target initial centers: [-1.5, -0.75, 0.0, 0.75, 1.5]
    # params[i] starts at 1, so offset_i = target_i - 1
    offsets = np.array([-2.5, -1.75, -1.0, -0.25, 0.5])
    
    # Extract center offsets and amplitudes
    center_offsets = params[0:5]
    amplitudes = params[5:10]
    
    # Compute actual centers
    centers = center_offsets + offsets
    
    # Reshape for broadcasting: x is (N,), centers is (5,)
    # We want to compute (x[:, None] - centers[None, :])
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    
    # Compute squared distances: (N, 5)
    diff = x_col - c_row
    sq_dist = diff * diff
    
    # Fixed width sigma = 0.5, so 2 * sigma^2 = 0.5
    # exp(-sq_dist / (2 * sigma^2)) = exp(-sq_dist / 0.5) = exp(-2 * sq_dist)
    kernels = np.exp(-2.0 * sq_dist)
    
    # Weighted sum: (N, 5) * (5,) -> (N,)
    prediction = kernels @ amplitudes
    
    return prediction
# EVOLVE-BLOCK-END
