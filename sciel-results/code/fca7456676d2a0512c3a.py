"""Program to find a,b with sigma(a)=sigma(b) minimizing |a/b - 3/2|, where error must be > 0."""
import math
import time

# EVOLVE-BLOCK-START
import time
import bisect
from collections import defaultdict
import numpy as np

MAX_N = 10_000_000
TIME_LIMIT = 85.0  # safety margin (evaluator kills at 90s)

def construct():
    t_start = time.time()
    deadline = t_start + TIME_LIMIT

    # --- Build sigma (divisor sum) via vectorized sieve up to MAX_N ---
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
    
    # Convert to list for faster scalar access
    sigma_list = sigma.tolist()

    # --- Build inverted index: sigma_value -> sorted list of numbers ---
    sigma_to_nums = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_to_nums[sigma_list[n]].append(n)
    
    # Sort the lists for binary search
    for key in sigma_to_nums:
        sigma_to_nums[key].sort()

    best_err = float('inf')
    best_pair = (1, 1)

    # --- Phase 1: Initial scan with fixed small window to establish baseline ---
    # This ensures best_err is not infinity before the dynamic phase
    W_INIT = 10
    for b in range(1, MAX_N + 1):
        if time.time() > deadline - 10:
            break
        sb = sigma_list[b]
        lst = sigma_to_nums[sb]
        if not lst:
            continue
        target = 1.5 * b
        start = bisect.bisect_left(lst, target - W_INIT)
        end = bisect.bisect_right(lst, target + W_INIT)
        
        for a in lst[start:end]:
            if a != b:
                err = abs(a / b - 1.5)
                if 0 < err < best_err:
                    best_err = err
                    best_pair = (a, b)

    # If no pair found (unlikely), set a safe bound
    if best_err == float('inf'):
        best_err = 1.0

    # --- Phase 2: Main scan with dynamic window, prioritizing large b ---
    # Iterate b from MAX_N down to 1
    # Window width w = int(b * best_err) + 2
    # This guarantees we don't miss any candidate that could improve best_err
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline - 1:
            break
        
        # Calculate dynamic window
        # We require |a/b - 1.5| < best_err
        # => |a - 1.5b| < b * best_err
        w = int(b * best_err) + 2
        if w < 1:
            w = 1
            
        sb = sigma_list[b]
        lst = sigma_to_nums[sb]
        if not lst:
            continue
            
        target = 1.5 * b
        start = bisect.bisect_left(lst, target - w)
        end = bisect.bisect_right(lst, target + w)
        
        for a in lst[start:end]:
            if a != b:
                err = abs(a / b - 1.5)
                if 0 < err < best_err:
                    best_err = err
                    best_pair = (a, b)

    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
