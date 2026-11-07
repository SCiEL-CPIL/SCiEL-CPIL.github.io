import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N using vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        return [1, 1]

    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # Phase 1: Vectorized search with a moderate window to find a good initial best_err
    # We check a in [floor(1.5*b) - 20, ceil(1.5*b) + 20]
    # This covers errors up to roughly 20/b. For b=1, err~20. For b=1000, err~0.02.
    # This should find a very small error quickly.
    
    b_arr = np.arange(1, N + 1, dtype=np.int64)
    a_centers = 1.5 * b_arr.astype(np.float64)
    
    # To avoid creating huge temporary arrays for each offset, we can do a few passes.
    # Or we can just check the 2 closest integers first, then expand if needed?
    # Let's check offsets -10 to 10. That's 21 passes.
    # 21 * 10^7 operations is ~2.1 * 10^8. Numpy can do this in < 1s.
    
    for offset in range(-10, 11):
        a_candidates = np.floor(a_centers + offset).astype(np.int64)
        valid_mask = (a_candidates >= 1) & (a_candidates <= N)
        if not np.any(valid_mask):
            continue
        a_valid = a_candidates[valid_mask]
        b_valid = b_arr[valid_mask]
        
        # Check sigma equality
        eq_mask = sigma[a_valid] == sigma[b_valid]
        if np.any(eq_mask):
            a_eq = a_valid[eq_mask]
            b_eq = b_valid[eq_mask]
            # Calculate errors: |2a - 3b| / (2b)
            # Use float64 for division
            errs = np.abs(2 * a_eq.astype(np.float64) - 3 * b_eq.astype(np.float64)) / (2.0 * b_eq.astype(np.float64))
            min_idx = np.argmin(errs)
            current_err = errs[min_idx]
            if current_err < best_err:
                best_err = current_err
                best_a = int(a_eq[min_idx])
                best_b = int(b_eq[min_idx])
                
    if time.time() > deadline:
        return [int(best_a), int(best_b)]
        
    # Phase 2: Python loop to polish with tight window
    # Window size W = 2 * best_err * b
    # If W < 1, we only check the closest integer.
    
    # We can also skip b if the theoretical min error is >= best_err
    # min_err(b) = min(|a/b - 1.5|) for integer a
    # a_closest = round(1.5*b)
    # min_err = |1.5*b - a_closest| / b
    
    for b in range(1, N + 1):
        if time.time() > deadline:
            break
            
        target = 1.5 * b
        # Theoretical min error
        a_closest = int(round(target))
        if a_closest < 1 or a_closest > N:
            continue
        min_possible_err = abs(target - a_closest) / b
        if min_possible_err >= best_err:
            continue
            
        # Define search window
        delta = best_err * b
        a_low = max(1, int(math.floor(target - delta)))
        a_high = min(N, int(math.ceil(target + delta)))
        
        # Iterate a in window
        for a in range(a_low, a_high + 1):
            if sigma[a] == sigma[b]:
                err = abs(2 * a - 3 * b) / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    # Update delta? It shrinks, so a_low/a_high might change?
                    # Since we are iterating a in a fixed range for this b, 
                    # and best_err only decreases, the condition err < best_err 
                    # will naturally filter out worse candidates.
                    # We don't need to recompute a_low/a_high inside the inner loop 
                    # because we already entered the range based on the *previous* best_err.
                    # However, if best_err decreases significantly, some 'a' in the current 
                    # range might now be outside the *new* tighter range? 
                    # No, we just check err < best_err. If it's not better, we ignore it.
                    # The loop continues to check other 'a's in the old range.
                    # This is safe, just slightly less efficient (checking some 'a's that 
                    # wouldn't have been in the tighter range). But the range is small.
                    
    return [int(best_a), int(best_b)]
