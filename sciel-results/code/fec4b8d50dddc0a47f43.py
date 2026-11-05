"""Low discrepancy on all arithmetic progressions - Multi-Start Iterated Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Returns (max_abs, worst_d, worst_i).
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    best_d = 1
    best_i = 0
    
    for d in range(1, n + 1):
        for i in range(min(d, n)):
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            cumsum = np.cumsum(seq)
            current_max = np.max(np.abs(cumsum))
            if current_max > max_abs:
                max_abs = current_max
                best_d = d
                best_i = i
    return int(max_abs), best_d, best_i

def _greedy_repair(signs, deadline, rng):
    """
    Perform greedy single-flip repairs until no improvement is found or deadline is reached.
    Returns the improved signs.
    """
    n = len(signs)
    current_disc, current_d, current_i = _get_max_ap_discrepancy(signs)
    
    while time.time() < deadline:
        # Identify indices in the worst AP
        ap_indices = list(range(current_i, n, current_d))
        
        if len(ap_indices) == 0:
            break
            
        best_flip_idx = -1
        best_flip_disc = current_disc
        best_flip_val = None
        
        # Evaluate flipping each index in the AP
        for idx in ap_indices:
            signs[idx] = -signs[idx]
            new_disc, _, _ = _get_max_ap_discrepancy(signs)
            
            if new_disc < best_flip_disc:
                best_flip_disc = new_disc
                best_flip_idx = idx
                best_flip_val = signs[idx]
            
            signs[idx] = -signs[idx]
        
        # Apply the best flip if found
        if best_flip_idx != -1 and best_flip_disc < current_disc:
            signs[best_flip_idx] = best_flip_val
            current_disc = best_flip_disc
            # Update worst AP for next iteration
            _, current_d, current_i = _get_max_ap_discrepancy(signs)
        else:
            # Local minimum reached for single flips in worst AP.
            # Try a few random flips to escape? 
            # In ILS, we usually stop repair and let the outer loop perturb.
            break
            
    return signs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    # Initial repair
    signs = _greedy_repair(signs, deadline, rng)
    
    best_signs = signs.copy()
    best_disc, _, _ = _get_max_ap_discrepancy(best_signs)
    
    # Iterated Local Search
    # Strategy:
    # 1. Perturb current best by flipping k random indices (k=3 to 5).
    # 2. Run greedy repair on the perturbed solution.
    # 3. If better than best, update best.
    # 4. Repeat until time limit.
    
    while time.time() < deadline - 5.0:  # Leave 5s for final checks
        # Perturb
        k = rng.integers(3, 6)  # 3 to 5 flips
        indices = rng.choice(n, size=k, replace=False)
        
        perturbed = best_signs.copy()
        perturbed[indices] = -perturbed[indices]
        
        # Repair
        repaired = _greedy_repair(perturbed, deadline, rng)
        
        # Evaluate
        disc, _, _ = _get_max_ap_discrepancy(repaired)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = repaired.copy()
        else:
            # Even if not better, if we are stuck (no improvement in last few iterations), 
            # we might want to try a larger perturbation or just continue.
            # For simplicity, we just continue with the current best.
            pass
            
    # Final check
    final_disc, _, _ = _get_max_ap_discrepancy(best_signs)
    if final_disc < best_disc:
        best_disc = final_disc
        # best_signs is already best_signs, but just in case
        pass

    return best_signs.tolist()

# EVOLVE-BLOCK-END
