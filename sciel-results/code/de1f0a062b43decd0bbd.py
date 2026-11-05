# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from itertools import product
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    n = 30
    mid = 15
    
    # Precompute unit fractions
    inv = [Fraction(1, i) for i in range(1, n + 1)]
    
    # Generate all combinations for first half
    first_half_sums = []
    for coeffs in product([-1, 0, 1], repeat=mid):
        s = Fraction(0)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * inv[i]
        first_half_sums.append((s, coeffs))
    
    # Generate all combinations for second half
    second_half_sums = []
    for coeffs in product([-1, 0, 1], repeat=n - mid):
        s = Fraction(0)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * inv[mid + i]
        second_half_sums.append((s, coeffs))
    
    # Sort second half by value for binary search
    second_half_sums.sort(key=lambda x: x[0])
    
    # Extract just the values for binary search
    second_half_values = [x[0] for x in second_half_sums]
    
    best_sum = None
    best_coeffs = None
    best_abs = None
    
    # For each first half sum, binary search for the closest second half sum
    for s1, coeffs1 in first_half_sums:
        target = -s1
        
        # Binary search for insertion point
        idx = bisect_left(second_half_values, target)
        
        # Check neighbors around the insertion point
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(second_half_values):
            candidates.append(idx)
        # Also check a few more neighbors
        for offset in [-2, -1, 0, 1, 2]:
            i = idx + offset
            if 0 <= i < len(second_half_values) and i not in candidates:
                candidates.append(i)
        
        for i in candidates:
            s2 = second_half_values[i]
            total = s1 + s2
            if total == 0:
                continue
            abs_val = abs(total)
            if best_abs is None or abs_val < best_abs:
                best_abs = abs_val
                best_sum = total
                best_coeffs = list(coeffs1) + list(second_half_sums[i][1])
    
    # Also try local search from the best found
    if best_coeffs is not None:
        curr = list(best_coeffs)
        curr_sum = best_sum
        curr_abs = best_abs
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            best_flip_idx = -1
            best_flip_val = None
            best_new_sum = None
            best_new_abs = curr_abs
            
            for i in range(n):
                for new_val in (-1, 0, 1):
                    if new_val == curr[i]:
                        continue
                    new_sum = curr_sum + (new_val - curr[i]) * inv[i]
                    if new_sum == 0:
                        continue
                    new_abs = abs(new_sum)
                    if new_abs < best_new_abs:
                        best_new_abs = new_abs
                        best_flip_idx = i
                        best_flip_val = new_val
                        best_new_sum = new_sum
            
            if best_flip_idx >= 0:
                curr[best_flip_idx] = best_flip_val
                curr_sum = best_new_sum
                curr_abs = best_new_abs
                improved = True
                if best_abs is None or curr_abs < best_abs:
                    best_sum = curr_sum
                    best_coeffs = list(curr)
                    best_abs = curr_abs
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
