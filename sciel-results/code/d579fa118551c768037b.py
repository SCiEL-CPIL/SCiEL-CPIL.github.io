"""Rewritten program: meet-in-the-middle with exact integer arithmetic via LCM.
Finds the subset of {1/2,...,1/48} with reciprocal sum strictly below 1,
minimizing the positive deficit 1-sum.
"""
# EVOLVE-BLOCK-START
import time

def construct():
    import time
    from fractions import Fraction
    from math import gcd
    from bisect import bisect_left, bisect_right

    deadline = time.time() + 80.0  # safety margin before 90s kill

    denominators = list(range(2, 49))  # 2..48, 47 items
    n = len(denominators)  # 47

    # Compute LCM of 2..48
    lcm_val = 1
    for d in denominators:
        lcm_val = lcm_val * d // gcd(lcm_val, d)

    # weights[i] = lcm_val // denominators[i], so sum of weights = sum of 1/d * lcm
    weights = [lcm_val // d for d in denominators]
    target = lcm_val  # we want sum strictly < target

    # Split into two halves
    mid = n // 2  # 23
    left_w = weights[:mid]
    right_w = weights[mid:]
    nl = mid
    nr = n - mid

    # Enumerate all subset sums of left half (exact ints)
    # 2^23 = 8,388,608
    left_sums = [0] * (1 << nl)
    for i in range(nl):
        w = left_w[i]
        bit = 1 << i
        for mask in range(bit):
            left_sums[bit | mask] = left_sums[mask] + w
    left_sums.sort()

    if time.time() > deadline - 5:
        # Fallback: greedy
        selected, total = [], 0
        for i, w in enumerate(weights):
            if total + w < target:
                selected.append(i)
                total += w
        return selected

    # For each subset sum of right half, find best left sum such that combined < target
    best_deficit = target  # deficit in units of 1/lcm; we want min positive deficit
    best_left_mask = None
    best_right_mask = None

    # Enumerate right half subset sums and masks
    nr_bits = 1 << nr
    # We'll iterate over right masks, compute sum, and binary search in left_sums
    # For speed, precompute right sums array
    right_sums = [0] * nr_bits
    for i in range(nr):
        w = right_w[i]
        bit = 1 << i
        for mask in range(bit):
            right_sums[bit | mask] = right_sums[mask] + w

    if time.time() > deadline - 3:
        selected, total = [], 0
        for i, w in enumerate(weights):
            if total + w < target:
                selected.append(i)
                total += w
        return selected

    import bisect
    for rmask in range(nr_bits):
        rsum = right_sums[rmask]
        if rsum >= target:
            continue
        # We need lsum < target - rsum
        limit = target - rsum  # lsum must be < limit, i.e., lsum <= limit - 1
        # Find largest lsum in left_sums that is <= limit - 1
        idx = bisect.bisect_left(left_sums, limit) - 1
        if idx < 0:
            continue
        lsum = left_sums[idx]
        combined = lsum + rsum
        if combined < target:
            deficit = target - combined
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = idx  # we need the actual mask, not just index
                best_right_mask = rmask
                # Early exit: deficit of 1 is optimal (smallest positive integer)
                if deficit == 1:
                    break

    if time.time() > deadline - 2:
        selected, total = [], 0
        for i, w in enumerate(weights):
            if total + w < target:
                selected.append(i)
                total += w
        return selected

    # Recover masks: best_left_mask is an index into sorted left_sums, not the mask.
    # We need to map back. Re-enumerate left to find mask with that sum value.
    # Better: store (sum, mask) pairs. Let's redo with mask tracking for the best only.
    # Since we only need one optimal solution, find a left mask achieving left_sums[best_left_mask].
    target_left_sum = left_sums[best_left_mask] if best_left_mask is not None else None

    # Find a left mask with sum == target_left_sum
    left_mask_found = None
    if target_left_sum is not None:
        for lmask in range(1 << nl):
            s = 0
            for i in range(nl):
                if lmask & (1 << i):
                    s += left_w[i]
            if s == target_left_sum:
                left_mask_found = lmask
                break

    selected = []
    if left_mask_found is not None and best_right_mask is not None:
        for i in range(nl):
            if left_mask_found & (1 << i):
                selected.append(i)
        for i in range(nr):
            if best_right_mask & (1 << i):
                selected.append(mid + i)

    if not selected:
        # Fallback greedy
        selected, total = [], 0
        for i, w in enumerate(weights):
            if total + w < target:
                selected.append(i)
                total += w

    return selected
# EVOLVE-BLOCK-END
