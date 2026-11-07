"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)

    # 10 Radial Basis Functions (Gaussian) centered at distinct points in [-3, 3]
    # Each RBF has a different width to allow local and global features
    # Centers are evenly spaced, widths vary to cover different scales
    
    # Centers: 10 points evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Widths: vary from narrow to wide
    # Narrow widths capture local features, wide widths capture global trends
    # Use a mix: some narrow, some medium, some wide
    widths = np.array([0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0])
    
    # Compute RBF basis: phi_j(x) = exp(-((x - c_j)^2) / (2 * sigma_j^2))
    # For numerical stability, we compute this vectorized
    # x has shape (n,), centers and widths have shape (10,)
    # We want basis[i, j] = exp(-((x[i] - centers[j])^2) / (2 * widths[j]^2))
    
    # Reshape for broadcasting: x -> (n, 1), centers -> (1, 10), widths -> (1, 10)
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    w_row = widths.reshape(1, -1)
    
    # Compute squared distances and scale by width
    diff = x_col - c_row  # (n, 10)
    sq_dist = diff * diff  # (n, 10)
    scaled = sq_dist / (2.0 * w_row * w_row)  # (n, 10)
    
    # Gaussian RBF
    basis = np.exp(-scaled)  # (n, 10)
    
    # Linear combination
    result = basis @ params
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    return result
# EVOLVE-BLOCK-END
