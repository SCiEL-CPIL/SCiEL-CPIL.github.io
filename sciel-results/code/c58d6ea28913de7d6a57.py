import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Start with seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Define a parameterized family
    # h_i = 1 + eps * cos(2 * pi * k * i / N)
    # We can also try h_i = 1 + eps * sin(...)
    
    def evaluate(h):
        s = np.sum(h)
        if s == 0:
            return float('inf')
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2 * N * max_conv / (s ** 2))
        return val
    
    # Grid search over k and eps
    # k is the harmonic number
    # eps is the amplitude
    
    # Try small eps first
    eps_candidates = np.linspace(-0.5, 0.5, 100)
    k_candidates = range(1, 20)
    
    for k in k_candidates:
        for eps in eps_candidates:
            if time.time() > deadline:
                break
            h = 1.0 + eps * np.cos(2 * np.pi * k * np.arange(N) / N)
            val = evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h
        
        if time.time() > deadline:
            break

    # Refine with scipy if time permits
    if time.time() < deadline - 10:
        import scipy.optimize as opt
        
        # Use the best found so far as initial guess
        # We need to map back to parameters
        # This is tricky if we don't store params.
        # Let's just do a local search on the best_h found
        
        # Actually, let's just do a few random perturbations of the best_h
        # Or assume the best was from the grid.
        
        # Let's try a specific known good candidate: 
        # h_i = 1 + eps * (-1)^i
        # This corresponds to k = N/2.
        
    return best_h
