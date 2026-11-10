"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy: Favor pairs that nearly fill the bin with a sharp tightness peak
    using squared inverse waste. We use min(d_i, d_j) as a scale factor instead
    of the product to avoid over-favoring large-large pairs which may waste
    space. This encourages a more balanced packing approach.
    
    H[i,j] = min(d_i, d_j) / (capacity - d_i - d_j + 1.0)^2  if fits
    H[i,j] = 1e-9                                          otherwise
    """
    start_time = time.time()
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    sums = d[:, None] + d[None, :]
    mins = np.minimum(d[:, None], d[None, :])
    fits = sums <= capacity
    
    waste = capacity - sums
    
    # Squared inverse waste with a floor of 1.0 to avoid extreme spikes
    # while still creating a strong preference for tight fits
    safe_waste = np.where(fits, waste + 1.0, 1.0)
    tightness = 1.0 / (safe_waste ** 2)
    
    raw_scores = mins * tightness
    H = np.where(fits, raw_scores, 1e-9)
    
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)
    
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
    
    elapsed = time.time() - start_time
    if elapsed > 325:
        pass
    
    return H
# EVOLVE-BLOCK-END
