"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

DENOMS = list(range(2, 49))
N = len(DENOMS)
RECIP = [Fraction(1, d) for d in DENOMS]

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s wall-clock kill

    # Precompute all subset sums via meet-in-the-middle
    half = N // 2
    left_denoms = DENOMS[:half]
    right_denoms = DENOMS[half:]
    left_recips = RECIP[:half]
    right_recips = RECIP[half:]

    # Enumerate all subsets of left half
    left_sums = []  # (sum, bitmask)
    n_left = len(left_denoms)
    for mask in range(1 << n_left):
        s = Fraction(0, 1)
        for i in range(n_left):
            if mask & (1 << i):
                s += left_recips[i]
                if s >= 1:
                    break
        if s < 1:
            left_sums.append((s, mask))

    if time.time() > DEADLINE:
        # Fallback: seed program
        selected, total = [], Fraction(0, 1)
        for i, d in enumerate(DENOMS):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    # Enumerate all subsets of right half
    right_sums = []  # (sum, bitmask)
    n_right = len(right_denoms)
    for mask in range(1 << n_right):
        s = Fraction(0, 1)
        for i in range(n_right):
            if mask & (1 << i):
                s += right_recips[i]
                if s >= 1:
                    break
        if s < 1:
            right_sums.append((s, mask))

    # Sort right_sums by sum descending for binary search
    right_sums.sort(key=lambda x: x[0], reverse=True)
    right_vals = [rs[0] for rs in right_sums]
    right_masks = [rs[1] for rs in right_sums]

    # For each left subset, find best right subset such that left + right < 1
    best_deficit = None
    best_left_mask = None
    best_right_mask = None

    for ls, lmask in left_sums:
        # Need rs such that ls + rs < 1, i.e., rs < 1 - ls
        target = 1 - ls  # strictly greater than this is not allowed; rs must be < target
        # Binary search for the largest rs < target
        # right_vals is sorted descending, so we want the last element that is < target
        # Actually, since sorted descending, we want the first element >= target, then take the one before it
        # Or: find the rightmost position where right_vals[i] < target
        
        lo, hi = 0, len(right_vals)
        # We want the largest index i such that right_vals[i] < target
        # Since sorted descending: right_vals[0] >= right_vals[1] >= ...
        # We want the last i where right_vals[i] < target
        
        # Binary search: find first position where right_vals[pos] < target
        # Then the best is at pos-1... no wait.
        # Since descending, right_vals[0] is largest.
        # We want max right_vals[i] such that right_vals[i] < target.
        # That's the element just before the first element that is >= target? No.
        # Let me think again. Sorted descending.
        # right_vals[0] >= right_vals[1] >= ... >= right_vals[n-1]
        # We want the largest value < target.
        # If right_vals[0] < target, then right_vals[0] is the answer.
        # If right_vals[0] >= target, we need to find where values drop below target.
        
        # Find the first index where right_vals[i] < target
        lo, hi = 0, len(right_vals)
        while lo < hi:
            mid = (lo + hi) // 2
            if right_vals[mid] < target:
                hi = mid
            else:
                lo = mid + 1
        # Now lo is the first index where right_vals[lo] < target
        # The best candidate is at index lo (smallest index with value < target, which has the largest value < target since descending)
        if lo < len(right_vals):
            rs = right_vals[lo]
            total = ls + rs
            deficit = 1 - total
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = lmask
                best_right_mask = right_masks[lo]
                if deficit == 0:
                    break  # Can't get better than 0 (but we need strict, so this won't happen with < check)

    if time.time() > DEADLINE or best_deficit is None:
        # Fallback: seed program
        selected, total = [], Fraction(0, 1)
        for i, d in enumerate(DENOMS):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    # Reconstruct indices
    selected = []
    for i in range(n_left):
        if best_left_mask & (1 << i):
            selected.append(i)
    for i in range(n_right):
        if best_right_mask & (1 << i):
            selected.append(half + i)
    
    if not selected:
        # Shouldn't happen, but fallback
        selected, total = [], Fraction(0, 1)
        for i, d in enumerate(DENOMS):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    return selected
# EVOLVE-BLOCK-END
