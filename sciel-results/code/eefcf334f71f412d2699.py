"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random
import numpy as np

_LCM = 232792560

def _find_collision_fast(subset):
    """Find two different subsets with the same sum, or None if dissociated.
    Returns (mask1, mask2) where mask1 < mask2."""
    k = len(subset)
    if k == 0:
        return None
    scaled = np.array([_LCM // x for x in subset], dtype=np.int64)
    n = 1 << k
    # Vectorized subset sum computation using DP: sums[mask] = sums[mask without lowest bit] + scaled[idx]
    # Build using the fact that mask's lowest set bit gives the element to add
    sums = np.zeros(n, dtype=np.int64)
    for i in range(k):
        bit = 1 << i
        # For all masks that have bit i set and where i is the lowest set bit
        # More efficient: iterate masks and use the recurrence
        # Alternative: use the standard DP where we build up
        pass
    # Standard DP approach vectorized by processing one bit at a time
    # Actually the cleanest vectorized approach:
    # sums[mask] = sum of scaled[i] for each i where mask has bit i
    # We can compute this as: for each i, add scaled[i] to all masks with bit i set
    for i in range(k):
        bit = 1 << i
        # All masks where bit i is set
        masks_with_bit = np.arange(bit, n, 2 * bit)  # masks: bit, bit+2*bit, ...
        # Wait, that's not right. Masks with bit i set are those where (mask & bit) != 0
        # These are: [bit, bit+1, ..., bit+bit-1], [bit+2*bit, ..., bit+2*bit+bit-1], etc.
        # Actually: masks where bit i is set form blocks of size bit separated by blocks of size bit
        # Range: [j*2*bit + bit, (j+1)*2*bit - 1] for j = 0, 1, ...
        # So: start = bit, end = bit + bit - 1, then next block starts at 2*bit + bit
        # Let's just use the recurrence which is simpler
        pass
    
    # Use the recurrence: for mask from 1 to n-1, find lowest set bit
    # This is hard to vectorize directly. Let's use a different approach:
    # Compute all subset sums via the property that we can build them iteratively
    # Start with sums[0] = 0
    # For each element i, for all existing masks, sums[mask | (1<<i)] = sums[mask] + scaled[i]
    # But we need to do this in a way that doesn't overwrite values we still need
    
    # Better: use the standard bit-DP which is O(n) but with a simple loop
    # For k <= 11, n <= 2048, this is fast enough even in Python
    # Let's use the original approach but with numpy for the final collision check
    
    sums = np.zeros(n, dtype=np.int64)
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    
    # Check for collisions using numpy
    # Find if any sum appears more than once
    # Use sorting approach
    sorted_sums, sort_indices = np.sort(sums), np.argsort(sums)
    # Check for consecutive equal values
    diffs = np.diff(sorted_sums)
    collision_pos = np.where(diffs == 0)[0]
    if len(collision_pos) > 0:
        pos = collision_pos[0]
        mask1 = int(sort_indices[pos])
        mask2 = int(sort_indices[pos + 1])
        return min(mask1, mask2), max(mask1, mask2)
    return None

def _greedy_remove(subset, heuristic='largest', deadline=None):
    """Start with subset, iteratively remove elements causing collisions."""
    current = list(subset)
    while True:
        if deadline and time.time() > deadline:
            break
        col = _find_collision_fast(current)
        if col is None:
            break
        mask1, mask2 = col
        k = len(current)
        sym_diff = []
        for i in range(k):
            bit = 1 << i
            if (mask1 & bit) != (mask2 & bit):
                sym_diff.append(i)
        if heuristic == 'largest':
            to_remove = max(sym_diff, key=lambda i: current[i])
        elif heuristic == 'smallest':
            to_remove = min(sym_diff, key=lambda i: current[i])
        else:
            to_remove = random.choice(sym_diff)
        current.remove(current[to_remove])
    return current

def _try_add_elements(best, removed, deadline):
    """Try to add back removed elements one by one."""
    current = list(best)
    for x in removed:
        if time.time() > deadline:
            break
        candidate = current + [x]
        candidate.sort()
        if _find_collision_fast(candidate) is None:
            current = candidate
    return current

def _try_swaps(best, not_in_set, deadline):
    """Try swapping elements in best with elements not in best."""
    current = list(best)
    in_set = set(current)
    remaining = [x for x in not_in_set if x not in in_set]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(current):
            if time.time() > deadline:
                break
            for b in remaining:
                candidate = current[:]
                candidate[i] = b
                candidate.sort()
                if _find_collision_fast(candidate) is None:
                    current = candidate
                    in_set = set(current)
                    remaining = [x for x in not_in_set if x not in in_set]
                    improved = True
                    break
    return current

def _multi_swap_search(best, not_in_set, deadline):
    """Try removing up to 2 elements and adding up to 2 from not_in_set."""
    current = list(best)
    in_set = set(current)
    remaining = [x for x in not_in_set if x not in in_set]
    best_found = list(current)
    
    # Try single swaps first (already done in _try_swaps, but do it here for consistency)
    for i, a in enumerate(current):
        if time.time() > deadline:
            break
        for b in remaining:
            candidate = current[:]
            candidate[i] = b
            candidate.sort()
            if _find_collision_fast(candidate) is None:
                if len(candidate) > len(best_found):
                    best_found = candidate
                else:
                    best_found = candidate
                in_set = set(best_found)
                remaining = [x for x in not_in_set if x not in in_set]
                break
    
    # Try double swaps: remove 2, add 2
    # Order remaining elements to try promising ones first
    # Prefer elements that are closer in magnitude to the ones being removed
    if time.time() < deadline - 5:
        n_cur = len(current)
        for i in range(n_cur):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_cur):
                if time.time() > deadline:
                    break
                # Sort remaining by proximity to the elements being removed
                removed_vals = [current[i], current[j]]
                remaining_sorted = sorted(remaining, key=lambda x: min(abs(x - r) for r in removed_vals))
                for b1 in remaining_sorted:
                    if time.time() > deadline:
                        break
                    for b2 in remaining_sorted:
                        if b2 == b1:
                            continue
                        candidate = current[:]
                        # Remove j first (higher index) to keep indices stable
                        candidate.pop(j)
                        candidate.pop(i)
                        candidate.extend([b1, b2])
                        candidate.sort()
                        if _find_collision_fast(candidate) is None:
                            if len(candidate) > len(best_found):
                                best_found = candidate
                                in_set = set(best_found)
                                remaining = [x for x in not_in_set if x not in in_set]
                            break
    return best_found

def construct():
    start_time = time.time()
    deadline = start_time + 82
    universe = list(range(1, 23))
    best = []
    best_len = 0
    heuristics = ['largest', 'smallest', 'random', 'random', 'largest', 'smallest', 'random']
    for h in heuristics:
        if time.time() > deadline - 15:
            break
        result = _greedy_remove(universe, heuristic=h, deadline=deadline - 15)
        removed = [x for x in universe if x not in result]
        result = _try_add_elements(result, removed, deadline - 10)
        not_in_set = [x for x in universe if x not in result]
        result = _try_swaps(result, not_in_set, deadline - 8)
        # Try multi-swap search
        if time.time() < deadline - 5:
            result2 = _multi_swap_search(result, not_in_set, deadline - 5)
            if len(result2) > len(result):
                result = result2
        if len(result) > best_len:
            best_len = len(result)
            best = result
    
    # Final polish: try adding back any missing elements
    in_set = set(best)
    not_in_set = [x for x in universe if x not in in_set]
    best = _try_add_elements(best, not_in_set, deadline - 2)
    # Try one more round of swaps with the improved set
    not_in_set = [x for x in universe if x not in set(best)]
    if time.time() < deadline - 1:
        best = _try_swaps(best, not_in_set, deadline - 1)
    return best
