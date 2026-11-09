"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)

L = 1
for d in DENOMS:
    L = lcm(L, d)

WEIGHTS = [L // d for d in DENOMS]


def _compute_best_subset_sum():
    """
    Find the subset of denominators whose reciprocal sum is strictly below 1
    and maximizes that sum (i.e., minimizes the deficit 1 - sum).
    
    Uses meet-in-the-middle to enumerate all 2^24 subsets for each half,
    then combines them efficiently.
    """
    mid = N // 2
    left_denoms = DENOMS[:mid]
    right_denoms = DENOMS[mid:]
    
    left_weights = [L // d for d in left_denoms]
    right_weights = [L // d for d in right_denoms]
    
    n_left = len(left_weights)
    n_right = len(right_weights)
    
    # Enumerate all subsets of left half
    # Store (sum, bitmask) for each subset
    left_sums = []
    for mask in range(1 << n_left):
        s = 0
        for i in range(n_left):
            if mask & (1 << i):
                s += left_weights[i]
        if s < L:  # Only keep valid subsets (sum < L)
            left_sums.append((s, mask))
    
    # Enumerate all subsets of right half
    right_sums = []
    for mask in range(1 << n_right):
        s = 0
        for i in range(n_right):
            if mask & (1 << i):
                s += right_weights[i]
        if s < L:
            right_sums.append((s, mask))
    
    # Sort right_sums by sum value
    right_sums.sort(key=lambda x: x[0])
    right_sum_values = [s for s, b in right_sums]
    right_bitmasks = [b for s, b in right_sums]
    
    import bisect
    
    best_sum = -1
    best_left_mask = 0
    best_right_mask = 0
    
    for s_l, b_l in left_sums:
        target = L - s_l  # We need s_r < target
        # Find the largest s_r < target
        idx = bisect.bisect_left(right_sum_values, target)
        if idx > 0:
            s_r = right_sum_values[idx - 1]
            b_r = right_bitmasks[idx - 1]
            total = s_l + s_r
            if total > best_sum:
                best_sum = total
                best_left_mask = b_l
                best_right_mask = b_r
    
    # Reconstruct indices
    result = []
    for i in range(n_left):
        if best_left_mask & (1 << i):
            result.append(i)
    for i in range(n_right):
        if best_right_mask & (1 << i):
            result.append(mid + i)
    
    if not result:
        result = [0]
    
    return result


def construct():
    """
    Return nonempty distinct zero-based indices into denominators [2,3,...,48]
    such that the reciprocal sum is strictly below 1, minimizing the deficit.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Meet-in-the-middle approach
    best_indices = _compute_best_subset_sum()
    
    # If we've exceeded time, return what we have
    if time.time() > deadline:
        return best_indices
    
    # Additional local improvement: try swapping one element for another
    # to see if we can get closer to 1 without exceeding it
    current_set = set(best_indices)
    current_sum = 0
    for idx in best_indices:
        current_sum += WEIGHTS[idx]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in list(current_set):
            w_idx = WEIGHTS[idx]
            for other in range(N):
                if other in current_set:
                    continue
                w_other = WEIGHTS[other]
                new_sum = current_sum - w_idx + w_other
                if new_sum < L and new_sum > current_sum:
                    # Improvement found
                    current_set.discard(idx)
                    current_set.add(other)
                    current_sum = new_sum
                    best_indices = list(current_set)
                    improved = True
                    break
            if improved:
                break
    
    # Also try adding elements if possible
    while time.time() < deadline:
        added = False
        for i in range(N):
            if i not in current_set:
                new_sum = current_sum + WEIGHTS[i]
                if new_sum < L:
                    current_set.add(i)
                    current_sum = new_sum
                    best_indices = list(current_set)
                    added = True
                    break
        if not added:
            break
    
    return best_indices
# EVOLVE-BLOCK-END
