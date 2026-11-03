"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct the step heights h to minimize the overlap at shift 5 (C5).
    
    Analysis:
    The score is described as an upper bound on C5, suggesting the metric
    depends heavily on the overlap at shift k=5.
    C(k) = sum_i h[i] * (1 - h[(i+k) % N]).
    For k=5, C(5) = sum_i h[i] - sum_i h[i]h[i+5] = 50 - <h, h_shifted>.
    To minimize C(5), we must maximize the inner product <h, h_shifted>.
    
    The maximum possible inner product is achieved when h is as "aligned" 
    with its shifted version as possible.
    If we use a block of 1s of length 50: h = [1]*50 + [0]*50.
    The shift by 5 overlaps the block with itself for 45 indices.
    <h, h_shifted> = 45.
    C(5) = 50 - 45 = 5.
    
    For the uniform distribution h = [0.5]*100:
    <h, h_shifted> = 100 * 0.25 = 25.
    C(5) = 50 - 25 = 25.
    
    5 is significantly lower than 25, leading to a higher score.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Construct the block solution
    h = np.zeros(N)
    h[:N//2] = 1.0
    
    # Verify constraints
    total = np.sum(h)
    if total != N / 2.0:
        # Fallback to uniform if construction fails (should not happen)
        h = np.full(N, 0.5)
        
    return h

# EVOLVE-BLOCK-END
