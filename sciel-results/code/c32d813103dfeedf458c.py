"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _validate(arr, n=12):
    """Check that arr has n distinct positive ints and all 2^n subset sums are distinct."""
    if len(arr) != n:
        return False
    if any(x <= 0 or not isinstance(x, int) for x in arr):
        return False
    if len(set(arr)) != n:
        return False
    # Compute all subset sums
    sums = [0]
    for x in arr:
        sums = sums + [s + x for s in sums]
    if len(sums) != (1 << n):
        return False
    if len(set(sums)) != (1 << n):
        return False
    return True

def _greedy_construct(n=12, max_val=1048576):
    """Greedy: pick smallest possible next element that keeps subset sums distinct."""
    arr = []
    # We'll maintain the set of all subset sums of the current arr
    sums = {0}
    for i in range(n):
        # Try candidates starting from 1
        for cand in range(1, max_val + 1):
            new_sums = sums | {s + cand for s in sums}
            if len(new_sums) == 2 * len(sums):
                # No collisions
                arr.append(cand)
                sums = new_sums
                break
        else:
            return None
    return arr

def _local_search(arr, n=12, max_val=1048576, max_iters=50000, time_limit=85.0):
    """Local search: try to reduce the max element by swapping elements."""
    start_time = time.time()
    best = arr[:]
    best_max = max(best)
    
    # Precompute subset sums for quick validation
    def subset_sums(a):
        ss = [0]
        for x in a:
            ss += [s + x for s in ss]
        return ss
    
    def is_valid(a):
        if len(a) != n:
            return False
        if any(x <= 0 for x in a):
            return False
        if len(set(a)) != n:
            return False
        ss = subset_sums(a)
        return len(set(ss)) == (1 << n)
    
    # Sort arr for consistency
    best = sorted(best)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random index to modify
        idx = random.randint(0, n - 1)
        old_val = best[idx]
        
        # Try to reduce this element
        # The new value must be different from all other elements
        other_vals = set(best[:idx] + best[idx+1:])
        
        # Try values from old_val - 1 down to 1, skipping values already in the array
        found = False
        for new_val in range(old_val - 1, 0, -1):
            if new_val in other_vals:
                continue
            trial = best[:idx] + [new_val] + best[idx+1:]
            if is_valid(trial):
                best = trial
                best_max = max(best)
                found = True
                break
        
        if not found:
            # Try a random perturbation: change one element to a random value
            if random.random() < 0.3:
                new_val = random.randint(1, max_val)
                if new_val not in other_vals:
                    trial = best[:idx] + [new_val] + best[idx+1:]
                    if is_valid(trial) and max(trial) <= best_max:
                        best = trial
                        best_max = max(best)
    
    return best

def construct():
    n = 12
    max_val = 1048576
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with greedy construction
    arr = _greedy_construct(n, max_val)
    if arr is None:
        # Fallback to powers of 2
        return [1 << i for i in range(n)]
    
    # Validate
    if not _validate(arr, n):
        return [1 << i for i in range(n)]
    
    best = arr
    best_max = max(best)
    
    # Local search to improve
    if time.time() < deadline - 5.0:
        improved = _local_search(best, n, max_val, max_iters=100000, time_limit=deadline - time.time() - 2.0)
        if improved is not None and max(improved) < best_max:
            best = improved
            best_max = max(best)
    
    # Final validation
    if not _validate(best, n):
        return [1 << i for i in range(n)]
    
    return best
# EVOLVE-BLOCK-END
