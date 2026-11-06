import time
import random
import numpy as np

def construct():
    N = 51975
    # Find all divisors of N greater than 1
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    if N > 1:
        divisors.append(N)
    divisors = sorted(set(divisors))
    
    # Precompute indices for each (m, r)
    indices = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            res_indices.append(np.arange(r, N, m))
        indices[m] = res_indices

    # Greedy Construction
    cover_count = np.zeros(N, dtype=np.int8)
    is_uncovered = np.ones(N, dtype=bool)
    uncovered_count = N
    
    # Sort divisors descending for greedy init
    sorted_divs = sorted(divisors, reverse=True)
    
    residues = {m: 0 for m in divisors}
    
    for m in sorted_divs:
        if uncovered_count == 0:
            break
            
        unc_idx = np.where(is_uncovered)[0]
        if len(unc_idx) == 0:
            break
            
        res = unc_idx % m
        counts = np.bincount(res, minlength=m)
        best_r = int(np.argmax(counts))
        best_gain = counts[best_r]
        
        if best_gain > 0:
            residues[m] = best_r
            idx = indices[m][best_r]
            
            was_uncovered = is_uncovered[idx]
            newly_covered = idx[was_uncovered]
            
            cover_count[idx] += 1
            
            if len(newly_covered) > 0:
                is_uncovered[newly_covered] = False
                uncovered_count -= len(newly_covered)

    # Initial best solution
    best_residues = residues.copy()
    best_cover_count = cover_count.copy()
    best_uncovered = uncovered_count
    
    current_residues = residues.copy()
    current_cover_count = cover_count.copy()
    current_uncovered = uncovered_count
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    num_divs = len(divisors)
    div_list = divisors
    
    # Local Search: Coordinate Descent with Best Response
    while time.time() < deadline:
        i = random.randint(0, num_divs - 1)
        m = div_list[i]
        old_r = current_residues[m]
        
        zero_mask = (current_cover_count == 0)
        if not np.any(zero_mask):
            break
            
        idx_zero = np.where(zero_mask)[0]
        res_zero = idx_zero % m
        counts_zero = np.bincount(res_zero, minlength=m)
        
        old_idx = indices[m][old_r]
        penalty = np.sum(current_cover_count[old_idx] == 1)
        
        best_new_r = int(np.argmax(counts_zero))
        
        dec = counts_zero[best_new_r]
        inc = penalty
        new_uncovered = current_uncovered + inc - dec
        
        if new_uncovered < current_uncovered:
            current_cover_count[old_idx] -= 1
            new_idx = indices[m][best_new_r]
            current_cover_count[new_idx] += 1
            current_residues[m] = best_new_r
            current_uncovered = new_uncovered
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = current_residues.copy()
                best_cover_count = current_cover_count.copy()
                
    # Format output
    result = []
    for m in div_list:
        result.append([m, best_residues[m]])
        
    return result
