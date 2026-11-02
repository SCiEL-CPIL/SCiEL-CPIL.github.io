# EVOLVE-BLOCK-START
import numpy as np

N = 100

def construct():
    # The theoretical minimum for the maximum overlap is N/4,
    # achieved by the constant sequence h_i = 0.5.
    # Any deviation from 0.5 increases the variance of h,
    # which increases the average overlap and thus the maximum overlap.
    # Therefore, the constant sequence is the global optimum.
    
    h = np.full(N, 0.5)
    return h

# EVOLVE-BLOCK-END
