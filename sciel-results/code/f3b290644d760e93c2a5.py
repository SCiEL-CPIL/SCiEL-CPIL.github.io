# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Learnable Radial Basis Function (RBF) Network.
    
    Unlike the fixed basis approach, this model uses 3 Gaussian "bumps" 
    whose centers, widths, and heights are determined by the 10 parameters.
    This allows the function to adapt its shape (sharp vs smooth, position 
    of features) to the hidden target, rather than just linearly combining 
    fixed shapes.
    
    Parameters mapping:
    p[0], p[1], p[2] : Centers of the 3 Gaussians
    p[3], p[4], p[5] : Widths (inverse variance) of the 3 Gaussians
    p[6], p[7], p[8] : Heights (amplitudes) of the 3 Gaussians
    p[9]             : Global offset (bias)
    """
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
        
    # Extract parameters
    centers = p[0:3]
    widths = p[3:6]
    heights = p[6:9]
    offset = p[9]
    
    # Ensure widths are positive to avoid division by zero or negative variance
    # Using tanh to map any real number to a positive range (0.1 to 100)
    # This makes the optimization landscape smoother for BFGS
    safe_widths = 0.1 + 99.9 * (np.tanh(widths) + 1.0) / 2.0
    
    # Compute the sum of Gaussians
    # result = sum(heights[i] * exp(-((x - centers[i])^2) / (2 * widths[i]^2))) + offset
    
    # Vectorized computation for efficiency
    # x shape: (N,)
    # centers shape: (3,)
    # We need to compute (x - c_i)^2 for all i and all x
    
    # Create a matrix of differences: (N, 3)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Squared differences: (N, 3)
    diff_sq = diff ** 2
    
    # Divide by 2 * width^2: (N, 3)
    # widths shape (3,) -> broadcast to (1, 3)
    denom = 2.0 * (safe_widths[np.newaxis, :] ** 2)
    
    # Exponentiate: (N, 3)
    exp_term = np.exp(-diff_sq / denom)
    
    # Weight by heights and sum: (N, 3) * (3,) -> (N, 3) -> sum(axis=1) -> (N,)
    weighted = exp_term * heights[np.newaxis, :]
    
    result = np.sum(weighted, axis=1) + offset
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return params[0] * x + params[1]
        
    return result
# EVOLVE-BLOCK-END
