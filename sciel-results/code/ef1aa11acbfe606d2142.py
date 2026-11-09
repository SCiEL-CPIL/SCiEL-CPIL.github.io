import numpy as np
import bisect

def construct():
    N = 10_000_000
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    # Create array of (sigma[n], n) for n=1..N
    # Sort by sigma[n]
    indices = np.arange(1, N + 1)
    sigma_vals = sigma[1:N+1]
    
    # Sort indices by sigma value
    order = np.argsort(sigma_vals)
    sorted_indices = indices[order]
    sorted_sigma = sigma_vals[order]
    
    # Find groups of equal sigma
    best_ratio = float('inf')
    best_a, best_b = None, None
    
    i = 0
    while i < N:
        j = i
        while j < N and sorted_sigma[j] == sorted_sigma[i]:
            j += 1
        # Group is sorted_indices[i:j]
        group_size = j - i
        if group_size >= 2:
            group = sorted_indices[i:j]
            # group is sorted by index (since we sorted by sigma, and for equal sigma, by index? No, argsort is stable but not necessarily by index)
            # Actually, we need to sort the group by value
            group = np.sort(group)
            
            # For each b in group, find a closest to 1.5*b
            for b in group:
                target = 1.5 * b
                # Binary search in group
                idx = bisect.bisect_left(group, int(target))
                for a_idx in [idx - 1, idx, idx + 1]:
                    if 0 <= a_idx < group_size:
                        a = group[a_idx]
                        if a != b:
                            ratio = abs(a / b - 1.5)
                            if ratio > 0 and ratio < best_ratio:
                                best_ratio = ratio
                                best_a, best_b = int(a), int(b)
        i = j
    
    return [best_a, best_b]
