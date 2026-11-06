import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Precompute all arithmetic progressions as index lists (0-based)
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, a+2s, ..., a+(k-1)s where a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # For efficient local search, precompute for each position i, which APs contain it
    # and the sign contribution
    # Actually, for flipping position i, the change in sum for AP p is:
    #   if i in p: delta = -2 * signs[i] (flipping from +1 to -1 or vice versa)
    #   else: delta = 0
    
    # Better approach: maintain current sums for all APs, and for each position,
    # know which APs it belongs to and its current contribution
    
    # Precompute: for each position i, list of (prog_id, sign_in_prog) 
    # Actually just prog_id, since sign is determined by current signs[i]
    
    pos_to_progs = [[] for _ in range(n)]
    for pid, idx in enumerate(progs):
        for i in idx:
            pos_to_progs[i].append(pid)
    
    def compute_sums(signs):
        sums = np.zeros(Np, dtype=np.int32)
        for pid, idx in enumerate(progs):
            s = 0
            for i in idx:
                s += signs[i]
            sums[pid] = s
        return sums
    
    def max_abs_sum(sums):
        return int(np.max(np.abs(sums)))
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n)
    sums = compute_sums(signs)
    best_signs = signs.copy()
    best_score = max_abs_sum(sums)
    
    # Local search: try flipping each position, keep if improves max abs sum
    # Use simulated annealing-like approach with time limit
    
    current_score = best_score
    
    # For efficiency, precompute the AP indices as numpy arrays
    prog_arrays = [np.array(idx, dtype=np.int32) for idx in progs]
    
    # Precompute for each position, the list of prog_ids it's in
    # and for quick sum updates
    
    # Local search with multiple restarts
    max_iter = 50000
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            pass  # already initialized
        else:
            signs = rng.choice([-1, 1], size=n)
            sums = compute_sums(signs)
            current_score = max_abs_sum(sums)
        
        if current_score < best_score:
            best_signs = signs.copy()
            best_score = current_score
        
        # Local search: iterate through positions, flip if it reduces max abs sum
        # Use a more efficient approach: for each flip, compute the new max
        # This is expensive, so use a greedy approach
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                
                # Try flipping position i
                old_sign = signs[i]
                new_sign = -old_sign
                signs[i] = new_sign
                
                # Update sums for all APs containing i
                for pid in pos_to_progs[i]:
                    sums[pid] += 2 * new_sign  # was old_sign, now new_sign, delta = new - old = -2*old = 2*new
                
                new_score = max_abs_sum(sums)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_signs = signs.copy()
                else:
                    # Revert
                    signs[i] = old_sign
                    for pid in pos_to_progs[i]:
                        sums[pid] -= 2 * new_sign  # revert: subtract 2*new_sign
    
    # Final check
    if time.time() < deadline - 2:
        # Try a few more targeted improvements
        for i in range(n):
            if time.time() > deadline - 2:
                break
            old_sign = signs[i]
            signs[i] = -old_sign
            for pid in pos_to_progs[i]:
                sums[pid] += 2 * signs[i]
            sc = max_abs_sum(sums)
            if sc < best_score:
                best_score = sc
                best_signs = signs.copy()
            else:
                signs[i] = old_sign
                for pid in pos_to_progs[i]:
                    sums[pid] -= 2 * signs[i]
    
    return [int(x) for x in best_signs]
