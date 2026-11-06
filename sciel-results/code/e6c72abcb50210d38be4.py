"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

_AP_MAT_CACHE = None
_COL_TO_ROWS_CACHE = None

def _build_ap_matrix(n=256):
    """
    Build a dense binary incidence matrix where each row corresponds to an
    arithmetic progression within [1, n] (1-indexed) and each column to a position.
    Entry (i, j) is 1 if position j+1 is in AP i.
    Also returns a list where col_to_rows[j] contains indices of APs containing column j.
    """
    aps = []
    for start in range(1, n + 1):
        for step in range(1, n - start + 1):
            # Generate the AP: start, start+step, ..., <= n
            ap_list = list(range(start, n + 1, step))
            if ap_list:
                aps.append(ap_list)
    
    m = len(aps)
    mat = np.zeros((m, n), dtype=np.int8)
    col_to_rows = [[] for _ in range(n)]
    
    for i, ap in enumerate(aps):
        # Vectorized assignment
        indices = np.array(ap, dtype=np.int32) - 1
        mat[i, indices] = 1
        for idx in indices:
            col_to_rows[idx].append(i)
            
    return mat, col_to_rows

def _get_cached_data(n=256):
    global _AP_MAT_CACHE, _COL_TO_ROWS_CACHE
    if _AP_MAT_CACHE is None:
        _AP_MAT_CACHE, _COL_TO_ROWS_CACHE = _build_ap_matrix(n)
    return _APAP_MAT_CACHE, _COL_TO_ROWS_CACHE

def _generate_initial_signs(n, rng, strategy):
    """Generate diverse initial sign patterns."""
    if strategy == 0:
        # Random
        return rng.choice([-1, 1], size=n).astype(np.float64)
    elif strategy == 1:
        # Alternating
        signs = np.ones(n, dtype=np.float64)
        signs[1::2] = -1
        return signs
    elif strategy == 2:
        # Thue-Morse
        signs = np.ones(n, dtype=np.float64)
        for i in range(n):
            if bin(i).count('1') % 2 == 1:
                signs[i] = -1
        return signs
    elif strategy == 3:
        # Sine-based
        x = np.linspace(0, 4*np.pi, n)
        signs = np.sign(np.sin(x))
        signs[signs == 0] = 1
        return signs
    elif strategy == 4:
        # Block pattern
        signs = np.ones(n, dtype=np.float64)
        for i in range(0, n, 8):
            signs[i:i+4] = -1
        return signs
    else:
        # Random with bias
        p = 0.3 + 0.4 * (strategy % 5) / 4
        return (2 * (rng.random(n) < p) - 1).astype(np.float64)

def _sa_run(ap_mat, col_to_rows, n, rng, time_limit, T_start, T_end, alpha, initial_signs=None):
    """
    Run a single instance of Simulated Annealing with incremental updates.
    """
    # Initial solution
    if initial_signs is not None:
        current_signs = initial_signs.copy()
    else:
        current_signs = rng.choice([-1, 1], size=n).astype(np.float64)
    
    # Initial sums
    current_sums = ap_mat @ current_signs
    current_cost = np.max(np.abs(current_sums))
    
    best_signs = current_signs.copy()
    best_cost = current_cost
    
    T = T_start
    start_time = time.time()
    
    # Track recent best to detect stagnation
    recent_best = current_cost
    stagnation_counter = 0
    
    while (time.time() - start_time) < time_limit and T > T_end:
        # Pick a random index to flip
        idx = rng.integers(0, n)
        
        old_sign = current_signs[idx]
        new_sign = -old_sign
        current_signs[idx] = new_sign
        
        # Update sums incrementally
        delta_sign = new_sign - old_sign  # +/-2
        affected_rows = col_to_rows[idx]
        
        if affected_rows:
            current_sums[affected_rows] += delta_sign
        
        # Calculate new cost
        new_cost = np.max(np.abs(current_sums))
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = current_signs.copy()
                recent_best = new_cost
                stagnation_counter = 0
        else:
            delta = new_cost - current_cost
            # Accept with probability exp(-delta / T)
            if delta > 0:
                # Optimization: if delta is large, exp(-delta/T) is tiny
                if delta / T > 50:
                    # Reject: revert
                    current_signs[idx] = old_sign
                    current_sums[affected_rows] -= delta_sign
                else:
                    if rng.random() < np.exp(-delta / T):
                        # Accept
                        if new_cost < recent_best:
                            recent_best = new_cost
                            stagnation_counter = 0
                        else:
                            stagnation_counter += 1
                    else:
                        # Reject: revert
                        current_signs[idx] = old_sign
                        current_sums[affected_rows] -= delta_sign
            else:
                # delta <= 0, always accept (already handled in if, but logic covers it)
                pass
        
        # Adaptive temperature boost when stuck
        if stagnation_counter > 1000:
            T = max(T, 5.0)
            stagnation_counter = 0
        
        T *= alpha
        
    return best_signs, best_cost

def _greedy_refine(ap_mat, col_to_rows, n, signs, sums, max_iters=10000):
    """
    Greedy refinement: repeatedly flip the sign that gives the largest improvement
    in the maximum absolute AP sum.
    """
    signs = signs.copy()
    sums = sums.copy()
    current_cost = np.max(np.abs(sums))
    
    for _ in range(max_iters):
        best_improvement = 0
        best_idx = -1
        
        # Try flipping each position and see if it improves the max sum
        for idx in range(n):
            old_sign = signs[idx]
            new_sign = -old_sign
            delta_sign = new_sign - old_sign  # +/-2
            
            # Compute new max sum efficiently
            affected_rows = col_to_rows[idx]
            if not affected_rows:
                continue
            
            # Temporarily update
            old_vals = sums[affected_rows].copy()
            sums[affected_rows] += delta_sign
            
            new_cost = np.max(np.abs(sums))
            
            # Revert
            sums[affected_rows] = old_vals
            
            improvement = current_cost - new_cost
            if improvement > best_improvement:
                best_improvement = improvement
                best_idx = idx
        
        if best_idx == -1:
            break
        
        # Apply the best flip
        old_sign = signs[best_idx]
        new_sign = -old_sign
        signs[best_idx] = new_sign
        delta_sign = new_sign - old_sign
        sums[col_to_rows[best_idx]] += delta_sign
        current_cost -= best_improvement
    
    return signs, sums, current_cost

def construct():
    n = 256
    ap_mat, col_to_rows = _get_cached_data(n)
    
    total_start = time.time()
    max_total_time = 55.0  # Leave 5s buffer for overhead
    
    rng = np.random.default_rng(42)
    
    best_global_signs = None
    best_global_cost = np.inf
    
    # Strategy: Multi-restart SA with diverse initializations
    strategies = [0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0]
    
    for r, strategy in enumerate(strategies):
        elapsed = time.time() - total_start
        remaining = max_total_time - elapsed
        if remaining <= 2.0:
            break
        
        # Allocate time: earlier restarts get less, later ones get more
        # But ensure each gets at least 2 seconds
        time_per_restart = min(remaining - 1.0, 5.0)
        
        initial_signs = _generate_initial_signs(n, rng, strategy)
        
        # Tune parameters based on restart
        T_start = 8.0 if strategy == 0 else 5.0
        T_end = 0.05
        alpha = 0.999
        
        best_signs, best_cost = _sa_run(ap_mat, col_to_rows, n, rng, time_per_restart, T_start, T_end, alpha, initial_signs)
        
        if best_cost < best_global_cost:
            best_global_cost = best_cost
            best_global_signs = best_signs.copy()
        
        if time.time() - total_start > max_total_time - 2.0:
            break
    
    if best_global_signs is None:
        best_global_signs = np.ones(n)
    
    # Final greedy refinement on the best solution
    best_sums = ap_mat @ best_global_signs
    best_global_signs, _, best_global_cost = _greedy_refine(ap_mat, col_to_rows, n, best_global_signs, best_sums, max_iters=10000)
    
    return [int(x) for x in best_global_signs]

# EVOLVE-BLOCK-END
