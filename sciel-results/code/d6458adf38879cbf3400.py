# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin

# Precompute the fixed centers for the 10 RBFs
# Evenly spaced in [-3, 3]
_RBF_CENTERS = np.linspace(-3.0, 3.0, 10)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-Grid Gaussian Process-like Representation.
    
    Strategy:
    Uses 10 fixed, evenly spaced RBF centers. 
    The 10 weights (p[0:10]) adapt to local features.
    A single shared width parameter (p[10] in a 11-param view, but we only have 10) 
    Wait, we have exactly 10 params. 
    Let's use p[0:9] as weights and p[9] as the log-width.
    This forces the model to have a consistent smoothness scale, which is a strong 
    inductive bias for functions that are not wildly multi-scale.
    
    Parameters (10 total):
    p[0] ... p[8] : Weights for the 10 fixed RBFs
    p[9]          : Log-width (log_sigma) shared by all RBFs
    """
    if time.time() > _DEADLINE:
        # Fallback: simple linear (uses first two params)
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
        
    # Extract weights and log-width
    weights = p[0:9]  # 9 weights? No, we have 10 centers. 
    # Let's re-allocate: 
    # If we use 10 centers, we need 10 weights. That leaves 0 params for width.
    # The prompt says "10 weights to adapt... while the width parameter is optimized".
    # This implies 11 parameters? But the contract says "array of length 10".
    # "Employ a non-linear Gaussian process-like representation using 10 fixed, evenly spaced Radial Basis Functions (RBFs) with a single shared, learnable width parameter, allowing the 10 weights to adapt..."
    # This sums to 11 parameters. But we only have 10.
    # Perhaps 9 weights + 1 width? Or 10 weights and fixed width?
    # "allowing the 10 weights to adapt ... while the width parameter is optimized"
    # This is a contradiction with the 10-param limit unless one weight is sacrificed or the width is not a param?
    # No, "width parameter is optimized by BFGS" means it must be in `params`.
    # So we must have 9 RBFs + 1 width? Or 10 RBFs where one weight is fixed?
    # Let's use 9 RBFs to fit the "10 weights" description loosely? 
    # Or better: Use 9 weights and 1 log-width. Total 10 params.
    # This gives 9 fixed centers.
    
    # Let's adjust centers to 9 for 9 weights + 1 width.
    # Actually, let's look at the inspiration: "10 fixed... RBFs... 10 weights... single shared width".
    # This is impossible with 10 params. 
    # I will implement 9 RBFs (weights) + 1 log-width. This is a valid structural change.
    
    # Re-define centers for 9 RBFs
    centers = np.linspace(-3.0, 3.0, 9)
    
    # p[0:9] are weights
    # p[9] is log_sigma
    w = p[0:9]
    log_sig = np.clip(p[9], -10, 10)
    sigma = np.exp(log_sig)
    
    # Calculate RBFs
    # x is (N, 1) or (N,). centers is (9,)
    # We need (N, 9) differences
    x_col = x.reshape(-1, 1)
    centers_row = centers.reshape(1, -1)
    
    diff = x_col - centers_row  # (N, 9)
    
    # exp(-0.5 * (diff/sigma)^2)
    z = diff / sigma
    rbf_values = np.exp(-0.5 * z * z)  # (N, 9)
    
    # Weighted sum
    result = rbf_values @ w  # (N,)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback: just the first few weights as a polynomial-like fallback?
        # Or return 0
        return np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
