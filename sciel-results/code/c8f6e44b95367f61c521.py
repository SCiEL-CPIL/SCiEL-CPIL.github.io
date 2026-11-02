N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # The uniform solution h[i] = 0.5 for all i is provably optimal:
    # overlap(k) = sum_i h[i]*(1-h[(i+k)%N]) = N/2 - autocorr(k)
    # Average autocorr over all k = N/4, so min autocorr <= N/4
    # Thus max overlap >= N/2 - N/4 = N/4
    # Uniform achieves exactly N/4 for all k, so it's optimal.
    h = np.full(N, 0.5)
    
    # Verify sum is exactly N/2
    s = np.sum(h)
    if abs(s - N/2.0) > 1e-12:
        h = h + (N/2.0 - s) / N
        h = np.clip(h, 0.0, 1.0)
    
    return h
# EVOLVE-BLOCK-END
