import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 30
    # Compute LCM(1, 2, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n+1))
    
    # For each i, the scaled term is L // i
    # We choose c_i in {-1, 0, 1}
    # Total sum = sum(c_i * L//i) / L
    # We want to minimize |sum(c_i * L//i)| with sum != 0
    
    # Meet in the middle: split into two halves of 15
    half = n // 2  # 15
    first_half = list(range(1, half+1))       # 1..15
    second_half = list(range(half+1, n+1))    # 16..30
    
    # Generate all possible sums for each half
    # For each element, coefficient is -1, 0, or 1
    # scaled_value[i] = L // i
    
    def gen_sums(indices):
        """Generate all 3^len(indices) possible sums"""
        vals = np.array([L // i for i in indices], dtype=np.int64)
        n_items = len(indices)
        # Use iterative construction
        # Start with [0], then for each item, triple the list
        sums = np.array([0], dtype=np.int64)
        for v in vals:
            # For each existing sum, create sum-v, sum, sum+v
            sums = np.concatenate([sums - v, sums, sums + v])
        return sums
    
    # Check deadline
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    sums1 = gen_sums(first_half)
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    sums2 = gen_sums(second_half)
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)
    
    # For each sum in sums1, find the best sum in sums2 such that:
    # total = s1 + s2 is nonzero and |total| is minimized
    # We want to minimize |s1 + s2| over all pairs, with s1+s2 != 0
    
    # For each s1, the best s2 is the one closest to -s1
    # Use searchsorted to find insertion point
    
    # We'll vectorize this as much as possible
    # For each s1, find the position in sums2_sorted where -s1 would be inserted
    insert_pos = np.searchsorted(sums2_sorted, -sums1, side='left')
    
    # Check candidates: insert_pos and insert_pos-1 (and maybe insert_pos+1)
    # Clamp to valid range
    n2 = len(sums2_sorted)
    
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    # We need to check a few candidates per s1
    # Let's check positions: insert_pos-1, insert_pos, insert_pos+1 (clamped)
    
    for offset in [-1, 0, 1]:
        pos = np.clip(insert_pos + offset, 0, n2-1)
        candidate_s2 = sums2_sorted[pos]
        total = sums1 + candidate_s2
        # Exclude zero
        nonzero_mask = (total != 0)
        if not np.any(nonzero_mask):
            continue
        abs_vals = np.abs(total[nonzero_mask])
        min_idx = np.argmin(abs_vals)
        min_val = abs_vals[min_idx]
        if best_abs is None or min_val < best_abs:
            best_abs = min_val
            # Find which s1 gave this
            nonzero_indices = np.where(nonzero_mask)[0]
            s1_idx = nonzero_indices[min_idx]
            best_s1 = sums1[s1_idx]
            best_s2 = candidate_s2[s1_idx]
    
    if best_abs is None:
        # Fallback
        return [0]*28 + [1, -1]
    
    # Now we need to recover the coefficients
    # We need to find which combination of c_i gives sum = best_s1 for first half
    # and which gives sum = best_s2 for second half
    
    def recover_coeffs(indices, target_sum):
        """Find coefficients in {-1,0,1} that give the target sum"""
        vals = [L // i for i in indices]
        n_items = len(indices)
        # Use iterative approach: try each coefficient
        coeffs = [0] * n_items
        remaining = target_sum
        # Backtracking from last to first
        for i in range(n_items - 1, -1, -1):
            v = vals[i]
            # Try 0, 1, -1
            found = False
            for c in [0, 1, -1]:
                new_remaining = remaining - c * v
                # Check if new_remaining is achievable with items 0..i-1
                # Max absolute sum with items 0..i-1
                max_possible = sum(vals[j] for j in range(i))
                if abs(new_remaining) <= max_possible:
                    # This is a necessary condition but not sufficient
                    # For now, just accept it (greedy might not work perfectly)
                    coeffs[i] = c
                    remaining = new_remaining
                    found = True
                    break
            if not found:
                # Try all options, pick one that seems best
                for c in [0, 1, -1]:
                    new_remaining = remaining - c * v
                    coeffs[i] = c
                    remaining = new_remaining
                    break
        # Verify
        total = sum(coeffs[i] * vals[i] for i in range(n_items))
        if total != target_sum:
            # Greedy failed, need proper search
            # Use recursive backtracking
            def backtrack(idx, rem):
                if idx == -1:
                    return rem == 0
                v = vals[idx]
                for c in [0, 1, -1]:
                    coeffs[idx] = c
                    if backtrack(idx - 1, rem - c * v):
                        return True
                return False
            if not backtrack(n_items - 1, target_sum):
                return None
        return coeffs
    
    coeffs1 = recover_coeffs(first_half, best_s1)
    if coeffs1 is None:
        return [0]*28 + [1, -1]
    
    coeffs2 = recover_coeffs(second_half, best_s2)
    if coeffs2 is None:
        return [0]*28 + [1, -1]
    
    result = coeffs1 + coeffs2
    assert len(result) == 30
    assert all(c in (-1, 0, 1) for c in result)
    
    # Verify the sum is nonzero
    total = sum(result[i] / (i+1) for i in range(30))
    if total == 0:
        return [0]*28 + [1, -1]
    
    return result
