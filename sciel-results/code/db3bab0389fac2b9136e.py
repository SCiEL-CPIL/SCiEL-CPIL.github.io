import time
from math import lcm
from itertools import combinations
import random

def _can_form_target_int(target_int, values_int, n_values):
    """
    Meet-in-the-middle: check if target_int can be formed as sum of a non-empty
    subset of values_int (all positive integers).
    """
    if n_values == 0:
        return False
    total = sum(values_int)
    if total < target_int:
        return False
    
    half = n_values // 2
    left = values_int[:half]
    right = values_int[half:]
    
    # Generate all subset sums for left half
    left_sums = set()
    for mask in range(1, 1 << len(left)):
        s = 0
        for i in range(len(left)):
            if mask & (1 << i):
                s += left[i]
        if s <= target_int:
            left_sums.add(s)
    
    # Generate all subset sums for right half and check
    for mask in range(1, 1 << len(right)):
        s = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
        if s > target_int:
            continue
        need = target_int - s
        if need == 0:
            return True
        if need in left_sums:
            return True
    
    # Also check if target can be formed using only left or only right
    if target_int in left_sums:
        return True
    
    # Check right-only subsets
    for mask in range(1, 1 << len(right)):
        s = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
        if s == target_int:
            return True
    
    return False

def _is_valid_set(S):
    S = sorted(S)
    n = len(S)
    # Compute LCM of all elements
    lcm_val = 1
    for x in S:
        lcm_val = lcm(lcm_val, x)
    
    for idx, x in enumerate(S):
        # Target: 1/x scaled by LCM = lcm_val // x
        target_int = lcm_val // x
        # Values: 1/y scaled by LCM = lcm_val // y for y != x
        values_int = [lcm_val // y for i, y in enumerate(S) if i != idx]
        if _can_form_target_int(target_int, values_int, n - 1):
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 85
    
    # Start with the known good set {13, 14, ..., 24} of size 12
    best = set(range(13, 25))
    best_sum = sum(best)
    
    # Local search with lookahead
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add an element (size + 1)
        added = False
        for c in range(1, 25):
            if c in best:
                continue
            trial = best | {c}
            if _is_valid_set(trial):
                best = trial
                best_sum = sum(best)
                improved = True
                added = True
                break
        
        if added:
            continue
        
        # Try swap with growth: remove d, add c, then add c' (size + 1)
        for c in range(1, 25):
            if c in best:
                continue
            for d in best:
                if time.time() > deadline:
                    break
                trial1 = best - {d} | {c}
                if _is_valid_set(trial1):
                    # Try to add another element to trial1
                    for c_prime in range(1, 25):
                        if c_prime in trial1:
                            continue
                        trial2 = trial1 | {c_prime}
                        if _is_valid_set(trial2):
                            best = trial2
                            best_sum = sum(best)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        
        if improved:
            continue

        # Try swap without growth but with better heuristic (larger sum)
        # This might open up paths for future growth
        for c in range(1, 25):
            if c in best:
                continue
            for d in best:
                if time.time() > deadline:
                    break
                trial = best - {d} | {c}
                if _is_valid_set(trial):
                    if sum(trial) > best_sum:
                        best = trial
                        best_sum = sum(best)
                        improved = True
                        break
            if improved:
                break

    return sorted(best)
