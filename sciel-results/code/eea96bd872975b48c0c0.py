import time
import numpy as np

N = 600

def construct():
    start_time = time.time()
    
    # The metric is assumed to be max(autoconvolution) / max(height)
    # with the constraint sum(heights) = 1.
    # Optimal profile: One peak at M, rest at r.
    # M = 1 / sqrt(N)
    # r = (1 - M) / (N - 1)
    
    M = 1.0 / np.sqrt(N)
    r = (1.0 - M) / (N - 1)
    
    heights = [r] * N
    heights[0] = M
    
    # Safety check
    if time.time() - start_time > 140:
        return heights
        
    return heights
