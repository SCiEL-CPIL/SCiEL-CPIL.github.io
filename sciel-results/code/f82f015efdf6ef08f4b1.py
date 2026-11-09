"""Construct 12 distinct positive integers with unique subset sums, minimizing the largest element."""
import time
import random
from itertools import combinations

def _subset_sums(arr):
    """Compute all subset sums of arr."""
    sums = [0]
    for x in arr:
        sums = sums + [s + x for s in sums]
    return sums

def _is_valid(arr):
    """Check if all subset sums are distinct and arr has 12 distinct positive integers."""
    if len(arr) != 12:
        return False
    if len(set(arr)) != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    sums = _subset_sums(arr)
    return len(sums) == 4096

def _random_superincreasing(max_val, seed=42):
    """Generate a random superincreasing sequence with max element <= max_val."""
    rng = random.Random(seed)
    arr = [1]
    for i in range(1, 12):
        current_sum = sum(arr)
        # Choose next element in [current_sum + 1, min(max_val, ...)]
        # For superincreasing: a_i > sum(a_0..a_{i-1})
        lo = current_sum + 1
        hi = max_val
        if lo > hi:
            return None
        # Bias towards smaller values to keep max small
        # But we need to leave room for remaining elements
        remaining = 11 - i
        # Minimum possible max if we keep going superincreasing:
        # a_i >= sum + 1, a_{i+1} >= 2*sum + 1, etc.
        # This is complex; just pick randomly but weighted towards lower
        if rng.random() < 0.7:
            # Pick from lower half
            mid = lo + (hi - lo) // 4
            x = rng.randint(lo, mid)
        else:
            x = rng.randint(lo, hi)
        arr.append(x)
    if arr[-1] <= max_val:
        return arr
    return None

def _local_search(arr, max_val, max_iters=5000, seed=42):
    """Try to reduce the maximum element via local search."""
    rng = random.Random(seed)
    arr = sorted(arr)
    best_max = arr[-1]
    
    for _ in range(max_iters):
        # Pick a random index to modify (not the first, to keep it small)
        idx = rng.randint(1, 11)
        old_val = arr[idx]
        
        # Try to decrease this value
        # The value must be > sum of all previous elements (for superincreasing property)
        # But we don't strictly require superincreasing, just unique subset sums
        # Let's try a range of smaller values
        prefix_sum = sum(arr[:idx])
        # Minimum valid value is prefix_sum + 1 for superincreasing
        # But even non-superincreasing could work if subset sums remain unique
        # Try values from max(1, prefix_sum + 1) up to old_val - 1
        lo = max(1, prefix_sum + 1)
        hi = old_val - 1
        
        if lo > hi:
            continue
        
        # Try a few random values in this range
        candidates = []
        if hi - lo < 20:
            candidates = list(range(hi, lo - 1, -1))  # Try largest first to reduce max
        else:
            # Try a few random values biased towards larger (closer to old_val)
            for _ in range(10):
                candidates.append(rng.randint(lo, hi))
            # Also try the largest few
            candidates.extend(range(min(hi, hi), max(lo, hi - 5), -1))
        
        for new_val in candidates:
            new_arr = arr[:idx] + [new_val] + arr[idx+1:]
            new_arr_sorted = sorted(new_arr)
            if new_arr_sorted[-1] >= best_max:
                continue
            if _is_valid(new_arr_sorted):
                arr = new_arr_sorted
                best_max = arr[-1]
                break
    
    return arr, best_max

def _greedy_construct(max_val):
    """Greedy construction: pick smallest valid value at each step."""
    arr = []
    current_sums = {0}
    
    for i in range(12):
        # Try values from 1 upwards
        for x in range(1, max_val + 1):
            if x in arr:
                continue
            # Check if adding x creates duplicate subset sums
            new_sums = current_sums | {s + x for s in current_sums}
            if len(new_sums) == len(current_sums) * 2:
                # All new sums are distinct from old
                arr.append(x)
                current_sums = new_sums
                break
        else:
            return None  # Couldn't find a valid value
    
    return arr

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # First, try greedy construction which should give powers of 2
    greedy_result = _greedy_construct(1048576)
    if greedy_result and _is_valid(greedy_result):
        best = greedy_result
        best_max = best[-1]
    else:
        best = [1 << i for i in range(12)]
        best_max = best[-1]
    
    # Try to find a better solution via local search
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try local search starting from powers of 2
        arr, new_max = _local_search([1 << i for i in range(12)], best_max, max_iters=2000, seed=123)
        if new_max < best_max:
            best = arr
            best_max = new_max
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Try more random restarts
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            rand_arr = _random_superincreasing(best_max, seed=seed + 1000)
            if rand_arr and _is_valid(rand_arr):
                if rand_arr[-1] < best_max:
                    best = rand_arr
                    best_max = rand_arr[-1]
                    # Try to improve further
                    if time.time() < deadline - 5:
                        arr2, new_max2 = _local_search(best, best_max, max_iters=1000, seed=seed + 2000)
                        if new_max2 < best_max:
                            best = arr2
                            best_max = new_max2
    
    return best
