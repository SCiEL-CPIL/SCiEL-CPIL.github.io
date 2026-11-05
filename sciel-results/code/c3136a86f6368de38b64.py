"""Rewritten program: meet-in-the-middle exhaustive search for minimal nonzero signed harmonic sum."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import product
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 30
    # Split indices 1..30 into two halves of 15
    half = n // 2  # 15
    
    # Build all possible sums for the first half (indices 1..15)
    # Each coefficient is in {-1, 0, 1}
    # sum = c[0]/1 + c[1]/2 + ... + c[14]/15
    # Use Fraction for exact arithmetic
    
    # Precompute denominators for each half
    left_indices = list(range(1, half + 1))       # 1..15
    right_indices = list(range(half + 1, n + 1))  # 16..30
    
    # Generate all 3^15 combinations for each half
    # For efficiency, store (sum_value, coeff_tuple) pairs
    
    left_sums = []
    for coeffs in product([-1, 0, 1], repeat=half):
        s = Fraction(0, 1)
        for i, idx in enumerate(left_indices):
            if coeffs[i] != 0:
                s += Fraction(coeffs[i], idx)
        left_sums.append((s, coeffs))
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple nonzero result
        return [0]*28 + [1, -1]
    
    right_sums = []
    for coeffs in product([-1, 0, 1], repeat=half):
        s = Fraction(0, 1)
        for i, idx in enumerate(right_indices):
            if coeffs[i] != 0:
                s += Fraction(coeffs[i], idx)
        right_sums.append((s, coeffs))
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [0]*28 + [1, -1]
    
    # Now we need to find pairs (L, R) from left_sums and right_sums
    # such that L + R != 0 and |L + R| is minimized.
    # 
    # Strategy: sort right_sums by their value. For each left sum L,
    # we want R such that L + R is close to 0 but not equal to 0.
    # So we want R close to -L.
    #
    # Sort right_sums by value, then for each L, binary search for -L
    # and check neighbors.
    
    right_sums_sorted = sorted(right_sums, key=lambda x: x[0])
    right_values = [r[0] for r in right_sums_sorted]
    
    import bisect
    
    best_abs = None
    best_coeff = None
    best_pair = None
    
    for L_val, L_coeffs in left_sums:
        target = -L_val
        
        # Binary search for target in right_values
        pos = bisect.bisect_left(right_values, target)
        
        # Check positions around pos (pos-2 to pos+2) to handle ties and near-misses
        candidates = set()
        for offset in range(-3, 4):
            idx = pos + offset
            if 0 <= idx < len(right_sums_sorted):
                candidates.add(idx)
        
        for idx in candidates:
            R_val, R_coeffs = right_sums_sorted[idx]
            total = L_val + R_val
            if total == 0:
                continue
            abs_total = abs(total)
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                # Combine coefficients
                full_coeffs = list(L_coeffs) + list(R_coeffs)
                best_coeff = full_coeffs
                best_pair = (L_val, R_val)
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
    
    if best_coeff is not None:
        return best_coeff
    else:
        # Fallback
        return [0]*28 + [1, -1]
# EVOLVE-BLOCK-END
