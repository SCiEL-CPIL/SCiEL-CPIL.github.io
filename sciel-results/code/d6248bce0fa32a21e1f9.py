"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Initial guess: Uniform
    h_best = np.ones(N)
    val_best = 2.0
    
    # Try Gaussian windows
    # h_i = exp(-i^2 / 2sigma^2)
    # Normalize sum to 1
    
    indices = np.arange(N)
    center = (N - 1) / 2.0
    
    sigmas = np.linspace(10, 200, 50)
    
    for sigma in sigmas:
        if time.time() > deadline:
            break
            
        h = np.exp(-((indices - center) ** 2) / (2 * sigma ** 2))
        s = np.sum(h)
        if s == 0:
            continue
        h = h / s
        
        c = np.convolve(h, h)
        val = abs(2 * N * np.max(c) / (s ** 2)) # s is 1 now, so just 2*N*max(c)
        val = 2 * N * np.max(c)
        
        if val < val_best:
            val_best = val
            h_best = h
    
    # Try Raised Cosine
    # w_i = cos^2(pi * (i - center) / N)
    for alpha in np.linspace(0.1, 1.0, 20):
        if time.time() > deadline:
            break
        h = np.cos(np.pi * (indices - center) * alpha / N) ** 2
        s = np.sum(h)
        if s == 0:
            continue
        h = h / s
        
        c = np.convolve(h, h)
        val = 2 * N * np.max(c)
        
        if val < val_best:
            val_best = val
            h_best = h
            
    return h_best
# EVOLVE-BLOCK-END
