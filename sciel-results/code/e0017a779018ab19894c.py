"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute centers for the 3x3 grid
# Centers at -1, 0, 1
centers_x = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
centers_y = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
sigma = 0.5
sigma_sq_2 = 2.0 * sigma * sigma

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard (though this function is very fast)
    # The evaluator calls this repeatedly during optimization.
    # We just need to return the prediction.
    
    # x and y are 1D arrays of length N
    # params is 1D array of length 10
    
    # Compute RBF values
    # x shape (N,), centers_x shape (9,) -> need broadcasting
    # dx shape (N, 9)
    dx = x[:, np.newaxis] - centers_x[np.newaxis, :]
    dy = y[:, np.newaxis] - centers_y[np.newaxis, :]
    
    r_sq = dx * dx + dy * dy
    
    # phi shape (N, 9)
    phi = np.exp(-r_sq / sigma_sq_2)
    
    # params[0:9] are weights for RBFs
    # params[9] is the constant offset
    pred = phi @ params[:9] + params[9]
    
    return pred
# EVOLVE-BLOCK-END
