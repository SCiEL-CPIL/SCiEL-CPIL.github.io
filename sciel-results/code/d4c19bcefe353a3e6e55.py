"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _all_subset_sums_distinct(arr):
    """Check that all 2^n subset sums are distinct using vectorized computation."""
    n = len(arr)
    if n == 0:
        return True
    arr = np.array(arr, dtype=np.int64)
    # Generate all subset sums using cumulative XOR trick
    # For each bit position, add arr[i] to half the sums
    sums = np.zeros(1 << n, dtype=np.int64)
    for i in range(n):
        # For subsets that include element i, add arr[i]
        # Bit i is set in masks where (mask >> i) & 1 == 1
        # This creates a repeating pattern of 2^i zeros, 2^i ones
        half = 1 << i
        block = 1 << (i + 1)
        for start in range(0, 1 << n, block):
            sums[start + half:start + block] += arr[i]
    # Check if all sums are distinct
    sorted_sums = np.sort(sums)
    return np.all(np.diff(sorted_sums) > 0)

def _max_element(arr):
    return max(arr)

def _greedy_construct(n=12, deadline=None):
    """Greedy: pick smallest possible value at each step that keeps subset sums distinct."""
    arr = []
    for i in range(n):
        if deadline is not None and time.time() > deadline:
            break
        # Try values starting from 1
        for val in range(1, 1048577):
            candidate = arr + [val]
            if _all_subset_sums_distinct(candidate):
                arr = candidate
                break
        else:
            break
    return arr

def _local_search(arr, time_limit=60.0, max_iters=100000):
    """Local search: try to reduce the largest element by swapping/replacing elements."""
    start_time = time.time()
    best = list(arr)
    best_max = _max_element(best)
    
    n = len(best)
    
    # Sort for consistency
    best.sort()
    
    iters = 0
    while iters < max_iters:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 3.0:
            break
        iters += 1
        
        # Try to reduce the largest element
        largest_idx = best.index(best_max)
        
        # Try replacing the largest element with a smaller value
        found_improvement = False
        # Try values from best_max - 1 down to 1, but skip many to save time
        step = max(1, (best_max - 1) // 50)
        for new_val in range(best_max - 1, 0, -step):
            if new_val in best:
                continue
            candidate = best[:]
            candidate[largest_idx] = new_val
            candidate.sort()
            if _all_subset_sums_distinct(candidate):
                new_max = _max_element(candidate)
                if new_max < best_max:
                    best = candidate
                    best_max = new_max
                    found_improvement = True
                    break
        
        if not found_improvement:
            # Try fine-grained search around current values
            for i in range(n):
                for offset in range(1, 20):
                    new_val = best[i] - offset
                    if new_val <= 0 or new_val in best:
                        continue
                    candidate = best[:]
                    candidate[i] = new_val
                    candidate.sort()
                    if _all_subset_sums_distinct(candidate):
                        new_max = _max_element(candidate)
                        if new_max <= best_max:
                            best = candidate
                            best_max = new_max
                            found_improvement = True
                            break
                if found_improvement:
                    break
        
        if not found_improvement:
            # Try random perturbation
            if random.random() < 0.5:
                candidate = best[:]
                num_changes = random.randint(1, 3)
                valid = True
                for _ in range(num_changes):
                    idx = random.randint(0, n - 1)
                    old_val = candidate[idx]
                    for _ in range(50):
                        new_val = random.randint(1, max(best_max, 100))
                        if new_val == old_val or new_val in candidate:
                            continue
                        candidate[idx] = new_val
                        break
                    else:
                        valid = False
                        break
                if valid:
                    candidate.sort()
                    if _all_subset_sums_distinct(candidate):
                        new_max = _max_element(candidate)
                        if new_max <= best_max:
                            best = candidate
                            best_max = new_max
                            found_improvement = True
    
    return best

def construct():
    """Construct 12 distinct positive integers <= 1048576 with all subset sums distinct, minimizing the largest."""
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit (leave 10s buffer)
    
    # Start with powers of 2 (known valid)
    best = [1 << i for i in range(12)]
    best_max = 2048
    
    # Try greedy construction
    if time.time() < deadline - 15.0:
        greedy = _greedy_construct(12, deadline=deadline - 15.0)
        if len(greedy) == 12:
            g_max = _max_element(greedy)
            if g_max < best_max:
                best = greedy
                best_max = g_max
    
    # Local search to improve
    if time.time() < deadline - 3.0:
        remaining = deadline - time.time()
        improved = _local_search(best, time_limit=remaining)
        if _max_element(improved) < best_max:
            best = improved
            best_max = _max_element(best)
    
    # Final validation
    if not _all_subset_sums_distinct(best):
        # Fallback to powers of 2
        return [1 << i for i in range(12)]
    
    return best
# EVOLVE-BLOCK-END
