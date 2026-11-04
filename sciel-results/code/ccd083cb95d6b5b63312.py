"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 50

# EVOLVE-BLOCK-START

def compute_ratio(h):
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
    l2_sq = np.sum(h**2)
    # Autoconvolution
    auto_conv = np.convolve(h, h, mode='full')
    max_peak = np.max(auto_conv)
    if max_peak < 1e-12:
        return -1.0
    return l2_sq / (l1 * max_peak)

def optimize_heights(support_indices):
    K = len(support_indices)
    if K == 0:
        return np.zeros(N)
    
    # Initial guess: uniform
    x0 = np.ones(K) / K
    
    def objective(x):
        # Normalize to sum to 1 to handle scale invariance
        s = np.sum(x)
        if s < 1e-12:
            return 1.0 # Bad
        x_norm = x / s
        h = np.zeros(N)
        h[support_indices] = x_norm
        return -compute_ratio(h)
    
    bounds = [(0, 1)] * K
    
    try:
        # Use Nelder-Mead for small K as it doesn't need gradients and handles non-smoothness better
        if K <= 15:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        else:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
            
        if res.success:
            x_opt = res.x
        else:
            # Fallback to best found or initial
            x_opt = res.x if hasattr(res, 'x') else x0
    except Exception:
        x_opt = x0
        
    # Ensure non-negative
    x_opt = np.maximum(x_opt, 0)
    
    # Construct final vector
    h = np.zeros(N)
    total = np.sum(x_opt)
    if total > 1e-12:
        h[support_indices] = x_opt / total
    else:
        h[support_indices] = 1.0 / K
        
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    best_h = np.ones(N)  # Default to seed (uniform)
    best_score = compute_ratio(best_h)
    
    # List of supports to try
    supports_to_try = []
    
    # 1. Powers of 2 (Guaranteed Sidon)
    powers_of_2 = [1, 2, 4, 8, 16, 32]
    supports_to_try.append(powers_of_2)
    
    # 2. Random sparse sets
    # Try various sparsities
    for k in range(2, 15):
        # Try a few random instances for each k
        for _ in range(10):
            if time.time() > deadline:
                break
            rand_indices = random.sample(range(N), k)
            supports_to_try.append(rand_indices)
            
    # 3. Structured sets: Arithmetic progressions
    for start in range(0, N, 3):
        for step in range(2, 10):
            indices = []
            for i in range(start, N, step):
                indices.append(i)
            if 2 <= len(indices) <= 12:
                supports_to_try.append(indices)
                
    # 4. "Sidon-like" greedy construction
    # Try to build a set with minimal sum collisions
    for _ in range(5):
        if time.time() > deadline:
            break
        # Random start
        current_set = []
        used_sums = set()
        for i in range(N):
            # Calculate new sums if we add i
            new_sums = {2*i}
            for j in current_set:
                new_sums.add(i+j)
            
            # Count collisions
            collisions = len(new_sums.intersection(used_sums))
            
            # Heuristic: add if no collisions, or few collisions
            # To keep it simple, only add if no collisions (strict Sidon)
            # But strict Sidon is limited. 
            # Let's allow some collisions but prefer low ones.
            # For now, let's just try to find a strict Sidon set of size up to 7
            if collisions == 0:
                used_sums.update(new_sums)
                current_set.append(i)
                if len(current_set) >= 7:
                    break
        if len(current_set) >= 2:
            supports_to_try.append(current_set)

    # Shuffle supports to diversify the search order? No, deterministic is fine for reproducibility if needed, but random is better for exploration.
    # We already have random ones.
    
    for support in supports_to_try:
        if time.time() > deadline:
            break
        
        # Optimize heights for this support
        h_candidate = optimize_heights(support)
        score = compute_ratio(h_candidate)
        
        if score > best_score:
            best_score = score
            best_h = h_candidate
            
    # Return as a list of floats
    return list(best_h)

# EVOLVE-BLOCK-END
