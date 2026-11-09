"""Combined program using a Radial Basis Function (RBF) basis for global approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0 # Safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        return np.zeros_like(x)

    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Define centers for RBFs.
    # We use a grid of centers to cover the [-1, 1]^2 domain.
    # 3x3 grid: (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)
    # We will map the 9 parameters p1..p9 to these centers.
    # p0 is the bias term.
    
    centers_x = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    centers_y = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    
    # Coefficients for the RBFs
    coeffs = p1[0:9] # p1 is a scalar? No, params is a flat array.
    # Wait, p1..p9 are scalars from unpacking.
    # Let's group them.
    rbf_coeffs = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9])

    # Compute distances from each point (x, y) to each center (cx, cy)
    # x and y are 1D arrays of length N.
    # centers_x and centers_y are 1D arrays of length 9.
    
    # To vectorize efficiently:
    # x[:, None] has shape (N, 1)
    # centers_x[None, :] has shape (1, 9)
    # dist_x = x[:, None] - centers_x[None, :] -> (N, 9)
    # dist_y = y[:, None] - centers_y[None, :] -> (N, 9)
    
    dist_x = x[:, None] - centers_x[None, :]
    dist_y = y[:, None] - centers_y[None, :]
    
    dist_sq = dist_x**2 + dist_y**2
    
    # Gaussian RBF: exp(-alpha * dist_sq)
    # We need to choose alpha. 
    # If alpha is too large, the basis functions are very localized (like interpolation).
    # If alpha is too small, they are very broad (like low-order polynomials).
    # A value around 1.0 to 5.0 is often reasonable for [-1, 1].
    # Let's use alpha = 2.0.
    alpha = 2.0
    
    rbf_values = np.exp(-alpha * dist_sq) # Shape (N, 9)
    
    # Linear combination of RBFs
    # rbf_values @ rbf_coeffs -> (N, 9) @ (9,) -> (N,)
    pred = p0 + rbf_values @ rbf_coeffs
    
    return pred
# EVOLVE-BLOCK-END
