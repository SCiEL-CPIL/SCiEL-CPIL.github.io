"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Initial Candidate: Truncated Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    
    # 2. Fast Sup-Norm Evaluation
    # We use a dense grid for initial search, then refine.
    # For speed, we'll use a moderate grid size for the main loop.
    n_points = 1024
    angles = 2 * np.pi * np.arange(n_points) / n_points
    zs = np.exp(1j * angles)
    
    def eval_sup_norm(coeffs):
        # coeffs is a numpy array of shape (24,)
        # Evaluate P(z) for all z in zs
        # P(z) = sum c_k z^k
        # We can use matrix multiplication: zs_power[k, j] = z_j^k
        # Pre-compute powers? No, zs is fixed.
        # powers = np.vander(zs, N=24, increasing=True) # shape (1024, 24)
        # vals = powers @ coeffs
        # return np.max(np.abs(vals))
        
        # More efficient: Horner's rule vectorized?
        # Or just precompute the power matrix once.
        pass

    # Pre-compute the power matrix for the fixed grid
    # z_j^k for j in 0..N-1, k in 0..23
    # Shape: (N, 24)
    powers = np.ones((n_points, 24), dtype=complex)
    for k in range(1, 24):
        powers[:, k] = powers[:, k-1] * zs
        
    def fast_score(coeffs):
        vals = powers @ coeffs
        return np.max(np.abs(vals))
    
    best_val = fast_score(best_coeffs) / math.sqrt(24)
    
    # 3. Simulated Annealing
    current_coeffs = best_coeffs.copy()
    current_val = fast_score(current_coeffs) / math.sqrt(24)
    
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = random.randint(0, 23)
        new_coeffs = current_coeffs.copy()
        new_coeffs[idx] = -new_coeffs[idx]
        
        new_val = fast_score(new_coeffs) / math.sqrt(24)
        
        delta = new_val - current_val
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_coeffs = new_coeffs
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_coeffs = current_coeffs.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: Reset to best if stuck?
            # For simplicity, just keep going with low T (greedy)
            
    return best_coeffs.astype(int).tolist()
# EVOLVE-BLOCK-END
