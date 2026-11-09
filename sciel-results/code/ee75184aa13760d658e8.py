"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random

_LCM = 232792560

def _find_collision_fast(subset):
    """Find two different subsets with the same sum, or None if dissociated.
    Returns (mask1, mask2) where mask1 < mask2."""
    k = len(subset)
    if k == 0:
        return None
    scaled = [_LCM // x for x in subset]
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    seen = {}
    for mask in range(n):
        s = sums[mask]
        if s in seen:
            return seen[s], mask
        seen[s] = mask
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

def construct():
    start_time = time.time()
    deadline = start_time + 85
    universe = list(range(1, 23))
    best = []
    best_len = 0
    heuristics = ['largest', 'smallest', 'random', 'random', 'largest']
    for h in heuristics:
        if time.time() > deadline - 10:
            break
        result = _greedy_remove(universe, heuristic=h, deadline=deadline - 10)
        removed = [x for x in universe if x not in result]
        result = _try_add_elements(result, removed, deadline - 5)
        not_in_set = [x for x in universe if x not in result]
        result = _try_swaps(result, not_in_set, deadline - 3)
        if len(result) > best_len:
            best_len = len(result)
            best = result
    in_set = set(best)
    not_in_set = [x for x in universe if x not in in_set]
    best = _try_add_elements(best, not_in_set, deadline - 1)
    return best
