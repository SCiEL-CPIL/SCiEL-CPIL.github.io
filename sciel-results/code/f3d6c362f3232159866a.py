"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 10.0
    
    # Base solution
    h = np.full(N, 0.5)
    
    # Try Gaussian perturbation to potentially reduce max overlap
    # in linear correlation scenarios, or explore landscape.
    sigma = 10.0
    x = np.arange(N)
    center = N / 2.0
    eps = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    eps -= np.mean(eps)
    eps_max = np.max(np.abs(eps))
    if eps_max > 0:
        eps = eps / eps_max
        
    best_h = h
    best_score = np.inf
    
    # We check both circular and linear max overlap, 
    # but return the h that minimizes the likely metric.
    # Given the ambiguity, we try to minimize the Circular Max Overlap 
    # primarily, as it's the standard Erdos metric.
    
    for amp in [0.05, 0.1, 0.2]:
        h_candidate = 0.5 + amp * eps
        if np.min(h_candidate) < 0 or np.max(h_candidate) > 1:
            continue
            
        # Correct sum
        diff = N / 2.0 - np.sum(h_candidate)
        idx = np.argmin(np.abs(h_candidate - 0.5))
        h_candidate[idx] += diff
        
        if np.min(h_candidate) < 0 or np.max(h_candidate) > 1:
            continue
            
        # Calculate Circular Max Overlap
        max_overlap = 0
        for k in range(N):
            shifted_h = np.roll(h_candidate, k)
            overlap = np.sum(h_candidate * (1 - shifted_h))
            if overlap > max_overlap:
                max_overlap = overlap
                
        if max_overlap < best_score:
            best_score = max_overlap
            best_h = h_candidate.copy()
            
        if time.time() - start_time > deadline:
            break

    return best_h
# EVOLVE-BLOCK-END
