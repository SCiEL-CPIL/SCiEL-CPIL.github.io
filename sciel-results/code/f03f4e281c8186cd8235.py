"""Optimized distinct subset sums with small largest element."""
# EVOLVE-BLOCK-START
import time
import itertools

def _all_subset_sums_distinct(arr):
    """Check if all subset sums are distinct. Returns True if distinct."""
    n = len(arr)
    sums = [0]
    for x in arr:
        # Add x to all existing sums
        new_sums = [s + x for s in sums]
        sums.extend(new_sums)
    # Check all distinct
    return len(set(sums)) == len(sums)

def _subset_sums(arr):
    """Return set of all subset sums."""
    n = len(arr)
    sums = {0}
    for x in arr:
        new_sums = {s + x for s in sums}
        sums |= new_sums
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    # Known good starting point: superincreasing (powers of 2)
    # But we want to minimize the max. 
    # Greedy: pick smallest possible next element
    # Start with 1, then greedily pick smallest that works
    
    # Greedy construction: a_1=1, a_{i+1} = smallest > a_i such that all subset sums remain distinct
    def greedy_construct(n):
        arr = [1]
        for i in range(1, n):
            # Try candidates starting from arr[-1]+1
            candidate = arr[-1] + 1
            while True:
                test_arr = arr + [candidate]
                if _all_subset_sums_distinct(test_arr):
                    arr.append(candidate)
                    break
                candidate += 1
            if time.time() > deadline:
                break
        return arr
    
    # Try greedy
    best = greedy_construct(12)
    best_max = max(best) if len(best) == 12 else float('inf')
    
    # If greedy didn't complete, fall back to powers of 2
    if len(best) < 12:
        best = [1 << i for i in range(12)]
        best_max = max(best)
    
    # Local search: try to reduce the maximum element
    # Strategy: for the largest element, try replacing it with smaller values
    # and check if the set still has distinct subset sums
    improved = list(best)
    improved_max = best_max
    
    while time.time() < deadline:
        # Find the largest element and try to reduce it
        # Sort indices by value descending
        idx_order = sorted(range(12), key=lambda i: -improved[i])
        found_improvement = False
        
        for idx in idx_order:
            if time.time() > deadline:
                break
            old_val = improved[idx]
            # Try smaller values (but must be distinct from others and positive)
            # Try values from old_val-1 down to 1
            other_vals = set(improved[j] for j in range(12) if j != idx)
            for new_val in range(old_val - 1, 0, -1):
                if new_val in other_vals:
                    continue
                improved[idx] = new_val
                if _all_subset_sums_distinct(improved):
                    found_improvement = True
                    break
                improved[idx] = old_val
            else:
                # No improvement found for this idx, restore
                improved[idx] = old_val
                continue
            
            if found_improvement:
                improved_max = max(improved)
                break
        
        if not found_improvement:
            # No single-element reduction works, try pair swaps
            found_pair_improvement = False
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i+1, 12):
                    if time.time() > deadline:
                        break
                    vi, vj = improved[i], improved[j]
                    # Try to reduce the larger one
                    if vj >= vi:
                        large_idx, small_idx = j, i
                    else:
                        large_idx, small_idx = i, j
                    large_val = improved[large_idx]
                    other_vals = set(improved[k] for k in range(12) if k != large_idx)
                    for new_val in range(large_val - 1, 0, -1):
                        if new_val in other_vals:
                            continue
                        improved[large_idx] = new_val
                        if _all_subset_sums_distinct(improved):
                            found_pair_improvement = True
                            break
                        improved[large_idx] = large_val
                    if found_pair_improvement:
                        improved_max = max(improved)
                        break
                if found_pair_improvement:
                    break
            
            if not found_pair_improvement:
                break  # Local minimum reached
    
    # Verify final result
    if not _all_subset_sums_distinct(improved):
        # Fallback to powers of 2
        improved = [1 << i for i in range(12)]
    
    return improved
# EVOLVE-BLOCK-END
