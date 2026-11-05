"""Low discrepancy on all arithmetic progressions - Multi-start Iterated Local Search with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Vectorized approach for speed.
    signs: numpy array of +1/-1.
    Returns max_abs_discrepancy.
    """
    n = len(signs)
    s = signs
    max_abs = 0
    
    # Pre-compute nothing, just loop over steps.
    # For n=256, this is fast enough if we avoid Python overhead where possible.
    # However, pure python loop over d and i is ~256*256/2 operations.
    # Let's optimize the inner loop.
    
    for d in range(1, n + 1):
        # For step d, starts are 0 .. min(d-1, n-1)
        max_start = min(d, n)
        for i in range(max_start):
            # Extract sequence s[i:n:d]
            # Instead of slicing and cumsum (which allocates), 
            # we can use a loop or a specialized check.
            # Given n=256, slicing is fine, but let's try to be smart.
            
            # A faster way to check max prefix sum without full array allocation 
            # if we were doing this in C, but in NumPy:
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            # cumsum is efficient in C
            cs = np.cumsum(seq)
            m = np.max(np.abs(cs))
            if m > max_abs:
                max_abs = m
                # Early exit optimization? 
                # If max_abs reaches the theoretical max for this step (len(seq)), 
                # it can't get worse for this specific AP, but other APs might be worse.
                # Actually, the global max is bounded by n. 
                # If max_abs == n, we can break all loops.
                if max_abs == n:
                    return n
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_disc = n + 1  # Initial worst case
    
    # Multi-start strategy
    # We will run several restarts. Each restart does a short local search.
    # Local search: Greedy flip of the element in the worst AP that reduces global discrepancy the most.
    
    num_restarts = 20  # We can adjust this based on time
    max_iters_per_restart = 50  # Keep short to allow many restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Initialize random signs
        signs = rng.choice([-1, 1], size=n).astype(np.int16)
        current_disc = _compute_max_ap_discrepancy(signs)
        
        # 2. Local Search Loop
        for _ in range(max_iters_per_restart):
            if time.time() > deadline - 2.0:
                break
                
            # Find worst AP to guide the flip
            # We need to find the specific AP that has the max discrepancy to target flips.
            # The previous function only returned the value. Let's make a version that returns the AP.
            
            # To save time, we can just do a random kick if we don't want to track the exact AP 
            # and rely on the global discrepancy drop. But greedy on the worst AP is stronger.
            
            # Let's use a helper that returns the AP indices.
            # For efficiency, we'll inline the search for the worst AP.
            
            max_abs_val = 0
            best_d = 1
            best_i = 0
            
            for d in range(1, n + 1):
                max_start = min(d, n)
                for i in range(max_start):
                    seq = signs[i:n:d]
                    if len(seq) == 0: continue
                    cs = np.cumsum(seq)
                    m = np.max(np.abs(cs))
                    if m > max_abs_val:
                        max_abs_val = m
                        best_d = d
                        best_i = i
            
            current_disc = max_abs_val
            
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
                
            # Identify indices in the worst AP
            ap_indices = list(range(best_i, n, best_d))
            
            # Try flipping each index in the worst AP
            best_flip_idx = -1
            best_flip_disc = current_disc
            
            # Optimization: Only check a subset of indices if the AP is very long?
            # For n=256, max AP length is 256 (d=1). 256 evaluations * ~1ms = 0.25s.
            # This is too slow for 50 iters * 20 restarts.
            # We need to be smarter.
            
            # Alternative: Randomly pick a few indices from the worst AP to flip?
            # Or use a steepest descent approximation?
            
            # Let's limit the number of flips we evaluate per iteration to keep it fast.
            # If we evaluate all, we might timeout.
            # Let's try evaluating all if len(ap_indices) < 50, else sample 20.
            
            if len(ap_indices) > 30:
                # Sample indices to check
                check_indices = rng.choice(ap_indices, size=20, replace=False)
            else:
                check_indices = ap_indices
                
            for idx in check_indices:
                signs[idx] = -signs[idx]
                new_disc = _compute_max_ap_discrepancy(signs)
                if new_disc < best_flip_disc:
                    best_flip_disc = new_disc
                    best_flip_idx = idx
                signs[idx] = -signs[idx] # Revert
                
            if best_flip_idx != -1:
                signs[best_flip_idx] = -signs[best_flip_idx]
                current_disc = best_flip_disc
            else:
                # Stuck in local minimum. Do a "kick" (random flip of 1-3 bits)
                # This allows escaping the local minimum.
                num_kicks = rng.integers(1, 4)
                for _ in range(num_kicks):
                    idx = rng.integers(0, n)
                    signs[idx] = -signs[idx]
                current_disc = _compute_max_ap_discrepancy(signs)
                # Don't immediately revert the kick, let the next iteration deal with it
                # Or, if the kick made it worse, we might want to revert? 
                # In ILS, we usually keep the kick to move to a new basin.
                
    # Final check
    if best_signs is None:
        # Fallback to random if something went wrong (shouldn't happen)
        best_signs = rng.choice([-1, 1], size=n).astype(np.int16)
        
    return best_signs.tolist()

# EVOLVE-BLOCK-END
