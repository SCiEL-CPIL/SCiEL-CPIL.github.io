import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    base = 51975
    
    # Find all divisors of base
    divisors = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    
    # Filter: m > 1, odd
    divisors = [m for m in divisors if m > 1 and m % 2 == 1]
    divisors.sort(reverse=True) # Try large moduli first? Or doesn't matter for greedy
    
    # Precompute masks for each modulus and residue
    # masks[m][r] is a boolean array of length base
    # To save memory and time, we can store indices
    # indices[m][r] is an array of indices i where i % m == r
    
    # We will store as list of lists
    # indices_by_m = { m: [ np.array([i for i in range(base) if i % m == r]) for r in range(m) ] }
    
    # This precomputation takes O(base * num_divisors) which is 51975 * 47 ~ 2.4M operations. Fast.
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            # i = r, r+m, r+2m, ...
            res_indices.append(np.arange(r, base, m))
        indices_by_m[m] = res_indices

    covered = np.zeros(base, dtype=np.uint8)
    available_moduli = set(divisors)
    classes = []
    
    best_uncovered = base
    best_classes = []
    
    while available_moduli:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_m = None
        best_r = None
        best_indices = None
        
        # Iterate through available moduli
        for m in available_moduli:
            res_indices_list = indices_by_m[m]
            for r in range(m):
                idx = res_indices_list[r]
                # Count uncovered
                # covered[idx] == 0
                gain = np.sum(covered[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    best_indices = idx
                    
        if best_gain <= 0:
            break
            
        # Commit
        classes.append([best_m, best_r])
        covered[best_indices] = 1
        available_moduli.remove(best_m)
        
        current_uncovered = np.sum(covered == 0)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_classes = [c[:] for c in classes]
            
        if best_uncovered == 0:
            break
            
    return best_classes
