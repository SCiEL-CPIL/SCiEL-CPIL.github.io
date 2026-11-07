"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2.
Strategy: Discrete assignment via greedy local search (pair swaps) on a binary
structure, then continuous refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    """Compute the raw value (lower is better)."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def project(h):
    """Project h to satisfy: h in [0,1]^N, sum(h) ~ N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h


def compute_correlation(h):
    """Compute full correlation of h and 1-h."""
    return np.correlate(h, 1.0 - h, mode="full")


def construct():
    """
    Discrete assignment strategy:
    1. Start with a binary pattern (50 zeros, 50 ones) arranged to minimize overlap.
    2. Use greedy local search (pair swaps) to optimize the binary arrangement.
    3. Refine the best binary pattern with continuous local optimization.
    
    The key insight: the optimal solution likely has a strong binary structure
    (values close to 0 or 1) because the objective is driven by max overlap,
    which is minimized when h and 1-h are as disjoint as possible.
    """
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0
    
    if remaining_time() < 5:
        return project(best_h)
    
    # Step 1: Generate good binary initial patterns
    # Pattern 1: Alternating 0,1,0,1,...
    # Pattern 2: Blocks of 0s and 1s
    # Pattern 3: Random binary with 50 zeros and 50 ones
    
    binary_patterns = []
    
    # Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    binary_patterns.append(alt.copy())
    
    # Block: first 50 are 0, next 50 are 1
    block = np.zeros(N)
    block[50:] = 1.0
    binary_patterns.append(block.copy())
    
    # Block: 25-25-25-25 pattern
    block2 = np.zeros(N)
    block2[25:50] = 1.0
    block2[75:] = 1.0
    binary_patterns.append(block2.copy())
    
    # Random binary patterns
    np.random.seed(42)
    for _ in range(10):
        idx = np.random.permutation(N)
        rand_bin = np.zeros(N)
        rand_bin[idx[:50]] = 1.0
        binary_patterns.append(rand_bin.copy())
    
    # Step 2: Greedy local search on binary patterns
    # For each pattern, try all pair swaps and keep the best improvement
    
    for pat_idx, pattern in enumerate(binary_patterns):
        if remaining_time() < 5:
            break
        
        current = pattern.copy()
        current_val = objective(current)
        
        # Greedy local search: repeatedly find the best swap
        improved = True
        max_iterations = 50
        iter_count = 0
        
        while improved and iter_count < max_iterations and remaining_time() > 5:
            improved = False
            iter_count += 1
            
            # Find the best pair swap that reduces the objective
            best_swap_val = current_val
            best_swap_pair = None
            best_swap = None
            
            # For efficiency, only try a subset of swaps
            # Try swaps involving positions where h is high or low
            # Actually, let's try all pairs but be smart about it
            
            # Precompute correlation for current
            corr = compute_correlation(current)
            
            # For each pair (i, j), compute the change in max correlation
            # This is expensive, so we'll sample pairs
            # Try pairs within a certain distance first, then random pairs
            
            # Strategy: try all pairs but break early if we find a good one
            # Actually, with N=100, there are ~5000 pairs. Let's try them all.
            
            # For speed, we'll compute the effect of swapping h[i] and h[j]
            # on the correlation
            
            # The correlation at lag k is sum_{i} h[i] * (1-h[i+k]) for valid i
            # Swapping h[i] and h[j] changes multiple correlation terms
            
            # This is complex, so let's just evaluate the full objective for
            # a random subset of swaps
            
            num_swaps_to_try = 200
            tried_swaps = set()
            
            for _ in range(num_swaps_to_try):
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j or (i, j) in tried_swaps or (j, i) in tried_swaps:
                    continue
                tried_swaps.add((i, j))
                
                # Swap
                new_h = current.copy()
                new_h[i], new_h[j] = new_h[j], new_h[i]
                
                new_val = objective(new_h)
                
                if new_val < best_swap_val - 1e-12:
                    best_swap_val = new_val
                    best_swap_pair = (i, j)
                    best_swap = new_h.copy()
                    improved = True
                    # Don't break; try more swaps to find the best one
            
            if improved and best_swap is not None:
                current = best_swap
                current_val = best_swap_val
        
        # After local search, check if this is better than our best
        if current_val < best_val:
            best_val = current_val
            best_h = current.copy()
    
    # Step 3: Continuous refinement of the best binary pattern
    # Use the best binary pattern as initial guess for continuous optimization
    
    if remaining_time() > 10:
        # Use Nelder-Mead for continuous refinement
        def obj_continuous(x):
            h_p = project(x)
            return objective(h_p)
        
        # Start from the best binary pattern
        h_init = best_h.copy()
        
        res = minimize(
            obj_continuous,
            h_init,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        h_res = project(res.x)
        val_res = objective(h_res)
        
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()
    
    # Step 4: Try a few more random restarts with continuous optimization
    # from binary-like initial conditions
    
    for trial in range(5):
        if remaining_time() < 10:
            break
        
        np.random.seed(1000 + trial)
        # Create a binary-like pattern with some noise
        idx = np.random.permutation(N)
        h_init = np.zeros(N)
        h_init[idx[:50]] = 1.0
        # Add small noise
        h_init += np.random.uniform(-0.1, 0.1, N)
        h_init = project(h_init)
        
        res = minimize(
            obj_continuous,
            h_init,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        h_res = project(res.x)
        val_res = objective(h_res)
        
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()
    
    # Final sanity check
    best_h = project(best_h)
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
