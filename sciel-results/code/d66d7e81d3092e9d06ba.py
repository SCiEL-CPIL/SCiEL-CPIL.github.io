# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        # Pad with zeros
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
    
    # Strategy: Non-linear RBF Network
    # 10 parameters total:
    # p[0]: Shared width parameter (sigma). We use log-sigma or direct sigma?
    # Direct sigma is non-linear in the loss landscape but BFGS handles it.
    # To ensure stability and avoid division by zero or extremely narrow peaks,
    # we can map the parameter to a positive value. 
    # Let's use sigma = exp(p[0]) to ensure sigma > 0.
    # The remaining 9 parameters are the weights for 9 RBF centers.
    # Why 9 centers? 10 params - 1 width = 9 weights.
    # Centers are evenly spaced in [-3, 3].
    
    # Calculate sigma from p[0]
    # Starting value for p[0] is 1.0 (from evaluator). exp(1) ~ 2.718.
    # This is a reasonable initial width for domain [-3,3].
    sigma = np.exp(p[0])
    
    # Define 9 fixed centers evenly spaced in [-3, 3]
    # np.linspace(-3, 3, 9) gives: -3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Compute RBF values
    # x shape: (N,)
    # centers shape: (9,)
    # We need a matrix of distances: |x_i - c_j|
    # dist_sq shape: (N, 9)
    
    # Efficient computation
    # x[:, np.newaxis] - centers[np.newaxis, :]
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    dist_sq = diff ** 2
    
    # RBF: exp(-dist_sq / (2 * sigma^2))
    # Ensure sigma is not too small to prevent overflow/underflow issues in exp
    # If sigma is very small, dist_sq / (2*sigma^2) is huge, exp(-huge) -> 0.
    # If sigma is very large, dist_sq / (2*sigma^2) is small, exp(-small) -> ~1.
    # This is numerically stable.
    
    inv_2_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    
    # If sigma is extremely large, inv_2_sigma_sq is ~0, RBFs become ~1 (constant basis).
    # If sigma is extremely small, RBFs become very narrow spikes.
    
    rbf_values = np.exp(-dist_sq * inv_2_sigma_sq)
    
    # Weights are p[1] through p[9]
    weights = p[1:10]
    
    # Linear combination of RBFs
    # rbf_values shape: (N, 9)
    # weights shape: (9,)
    # result shape: (N,)
    result = rbf_values @ weights
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback if numerical instability occurs
        return params[0] * x + params[1]
    
    return result
# EVOLVE-BLOCK-END
