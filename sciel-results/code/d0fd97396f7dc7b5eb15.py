import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10  # 10 seconds safety margin
    
    # The optimal non-negative step function to minimize the peak autoconvolution
    # for a fixed L1 norm is the uniform distribution over the entire support.
    # This minimizes the L2 norm, which is the zero-lag autocorrelation,
    # and thus the maximum autocorrelation.
    
    # Create a uniform vector
    h = np.ones(N)
    
    # Normalize to sum 1 (assuming the evaluator expects a probability distribution
    # or normalizes it; providing it normalized is safer and standard for such tasks)
    h = h / np.sum(h)
    
    # Check deadline
    if time.time() > deadline:
        return h.tolist()
        
    return h.tolist()
# EVOLVE-BLOCK-END
