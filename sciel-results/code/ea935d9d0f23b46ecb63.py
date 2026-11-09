"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with 5 centers.
    
    We use 5 fixed 2D centers to cover the domain.
    The 10 parameters are split into:
    - params[0:5]: Weights for each RBF
    - params[5:10]: Log-gammas (inverse variances) for each RBF
    
    Using log-gammas ensures the gamma values are positive, which is required
    for Gaussian kernels. This allows the model to adapt the width of each
    kernel independently, providing a much more flexible hypothesis space
    than a fixed gamma.
    
    Centers are chosen to cover the [-1, 1] x [-1, 1] square:
    1. (0, 0)   - Center
    2. (-1, -1) - Bottom-Left Corner
    3. (1, -1)  - Bottom-Right Corner
    4. (-1, 1)  - Top-Left Corner
    5. (1, 1)   - Top-Right Corner
    """
    
    # Define 5 fixed centers
    centers = np.array([
        [ 0.0,  0.0],
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0,  1.0]
    ])
    
    # Split parameters
    weights = params[0:5]
    log_gammas = params[5:10]
    
    # Ensure gammas are positive and reasonably bounded
    # exp(log_gamma) can be very large or small.
    # Clipping log_gammas helps stability if optimizer wanders.
    # However, BFGS is usually well-behaved.
    # We can just use exp.
    gammas = np.exp(log_gammas)
    
    # x and y are arrays of shape (N,)
    # centers is shape (5, 2)
    
    # Compute squared distances
    # x shape (N, 1), centers[:, 0] shape (5, 1) -> diff (N, 5)
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    
    # RBF values: shape (N, 5)
    # gamma shape (5,) -> (1, 5) for broadcasting
    rbf_values = np.exp(-gammas[np.newaxis, :] * dist_sq)
    
    # pred shape (N,)
    pred = rbf_values @ weights
    
    return pred
# EVOLVE-BLOCK-END
