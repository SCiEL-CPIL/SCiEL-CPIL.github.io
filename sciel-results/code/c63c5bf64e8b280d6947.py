"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    Returns an array of overlap values C_k for k != 0.
    C_k = sum_i h_i * (1 - h_{i+k})
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return float(np.max(overlaps))

def _swap_pair_delta(h, i, j):
    """
    Compute the change in max overlap when swapping h[i] and h[j].
    For efficiency, we compute the new overlaps and return the new max.
    """
    if h[i] == h[j]:
        return None  # No change
    
    h_new = h.copy()
    h_new[i], h_new[j] = h[j], h[i]
    new_max = _max_overlap(h_new)
    return new_max, h_new

def _local_search(h, max_iters=5000, deadline=None):
    """
    Greedy local search: repeatedly find the best single swap that reduces max overlap.
    """
    current_max = _max_overlap(h)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        best_swap = None
        best_new_max = current_max
        
        # Try all pairs - O(N^2) per iteration
        # For N=100, that's 4950 pairs
        for i in range(N):
            for j in range(i+1, N):
                if h[i] == h[j]:
                    continue
                # Quick check: swap and compute
                h[i], h[j] = h[j], h[i]
                new_max = _max_overlap(h)
                h[i], h[j] = h[j], h[i]
                
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_swap = (i, j)
        
        if best_swap is not None:
            i, j = best_swap
            h[i], h[j] = h[j], h[i]
            current_max = best_new_max
            improved = True
        
        iter_count += 1
    
    return h, current_max

def _simulated_annealing(h, start_temp=0.05, end_temp=0.001, 
                          max_iters=50000, deadline=None, rng=None):
    """
    Simulated annealing with swap moves to minimize max overlap.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = h.copy()
    current_max = _max_overlap(current)
    best = current.copy()
    best_max = current_max
    
    n_iter = 0
    while n_iter < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule (geometric cooling)
        progress = n_iter / max_iters
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Random swap
        i, j = rng.choice(N, size=2, replace=False)
        if current[i] == current[j]:
            n_iter += 1
            continue
        
        # Perform swap
        current[i], current[j] = current[j], current[i]
        new_max = _max_overlap(current)
        
        delta = new_max - current_max
        
        # Accept if better or with probability exp(-delta/temp)
        if delta < 0 or (temp > 0 and np.log(rng.random()) < -delta / temp):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best = current.copy()
        else:
            # Reject: swap back
            current[i], current[j] = current[j], current[i]
        
        n_iter += 1
    
    return best, best_max

def _generate_initial_candidates(rng):
    """Generate diverse initial candidates."""
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Alternating patterns
    for period in [2, 3, 4, 5]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = 0.5 + 0.3 * np.cos(2 * np.pi * i / period)
        h = np.clip(h, 0, 1)
        candidates.append(h.copy())
    
    # Sinusoidal
    t = np.arange(N)
    for freq in range(1, 15):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(np.clip(trial, 0, 1).copy())
    
    # Random
    for _ in range(30):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand.copy())
    
    return candidates

def _adjust_sum(h, target=TARGET_SUM, tol=1e-12):
    """
    Adjust h so that sum(h) == target, keeping values in [0,1].
    Uses iterative redistribution.
    """
    h = np.clip(h, 0, 1).copy()
    for _ in range(200):
        diff = target - np.sum(h)
        if abs(diff) < tol:
            break
        
        if diff > 0:
            mask = h < 1.0
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] = np.minimum(h[mask] + step, 1.0)
        else:
            mask = h > 0.0
            if not np.any(mask):
                break
            step = (-diff) / np.sum(mask)
            h[mask] = np.maximum(h[mask] - step, 0.0)
    
    return h

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Uses simulated annealing with swap moves followed by local search.
    The discrete swap structure avoids the plateau issues of continuous optimization.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate and evaluate initial candidates
    candidates = _generate_initial_candidates(rng)
    
    best_h = None
    best_score = float('inf')
    
    for h_cand in candidates:
        h_adj = _adjust_sum(h_cand)
        score = _max_overlap(h_adj)
        if score < best_score:
            best_score = score
            best_h = h_adj.copy()
    
    # Multi-start simulated annealing + local search
    n_starts = 8
    for start_idx in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        if start_idx == 0:
            h_init = best_h.copy()
        else:
            h_init = rng.uniform(0.0, 1.0, size=N)
            h_init = _adjust_sum(h_init)
        
        # Simulated annealing
        sa_deadline = min(deadline - 20.0, start_time + 10.0 + 8.0 * start_idx)
        if start_idx == 0:
            sa_deadline = min(deadline - 20.0, start_time + 15.0)
        elif start_idx < 3:
            sa_deadline = min(deadline - 20.0, start_time + 12.0 + 2.0 * start_idx)
        else:
            sa_deadline = min(deadline - 20.0, start_time + 10.0 + 1.5 * start_idx)
        
        h_sa, sa_score = _simulated_annealing(
            h_init, 
            start_temp=0.03 + 0.01 * start_idx,
            end_temp=0.0005,
            max_iters=30000,
            deadline=sa_deadline,
            rng=rng
        )
        
        # Local search polish
        h_ls, ls_score = _local_search(h_sa, max_iters=2000, deadline=deadline - 10.0)
        
        if ls_score < best_score:
            best_score = ls_score
            best_h = h_ls.copy()
    
    # Final local search polish on best solution
    if time.time() < deadline - 5.0:
        h_final, final_score = _local_search(best_h, max_iters=5000, deadline=deadline - 2.0)
        if final_score < best_score:
            best_score = final_score
            best_h = h_final.copy()
    
    # Ensure exact sum
    best_h = _adjust_sum(best_h)
    
    # Final validity check
    assert np.all(best_h >= 0) and np.all(best_h <= 1), "Values out of [0,1]"
    assert abs(np.sum(best_h) - TARGET_SUM) < 1e-10, f"Sum {np.sum(best_h)} != {TARGET_SUM}"
    
    return best_h

# EVOLVE-BLOCK-END
