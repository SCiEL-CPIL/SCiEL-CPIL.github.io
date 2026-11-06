"""Program for large reciprocal-subsum-free sets using optimized greedy construction."""
import time
from math import lcm as _math_lcm
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


# Precompute LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = _math_lcm(_L, i)

# Precompute integer reciprocals: R[x] = _L // x
_R = [0] * 25
for i in range(1, 25):
    _R[i] = _L // i


def _subset_sum_exists(target, elements):
    """
    Check if target can be formed by a non-empty subset of elements.
    elements is a list of integers (scaled reciprocals).
    target is an integer (scaled reciprocal).
    Uses meet-in-the-middle for speed.
    """
    if not elements:
        return False
    if target <= 0:
        return False

    n = len(elements)
    if n == 0:
        return False

    # If target is larger than sum of all elements, impossible
    if target > sum(elements):
        return False

    # If any single element equals target, we found a subset of size 1
    for e in elements:
        if e == target:
            return True

    # For small n, just do brute force with early exit
    if n <= 10:
        # Brute force with bit masks
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
                    if s > target:
                        break
            if s == target:
                return True
        return False

    # Meet in the middle
    mid = n // 2
    left = elements[:mid]
    right = elements[mid:]

    def gen_sums(arr):
        sums = {0}
        for val in arr:
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns <= target:  # Only keep sums <= target
                    new_sums.add(ns)
            sums |= new_sums
        return sums

    left_sums = gen_sums(left)
    right_sums = gen_sums(right)

    # Check if any combination works
    for l_val in left_sums:
        r_val = target - l_val
        if r_val in right_sums:
            return True

    return False


def _is_valid(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each element x in S, 1/x must NOT equal the sum of reciprocals of any subset of S\{x}.
    """
    if not S:
        return True
    
    S_set = set(S)
    for x in S:
        target = _R[x]
        others = [_R[y] for y in S if y != x]
        if _subset_sum_exists(target, others):
            return False
    return True


def _greedy_construct(order):
    """
    Greedy construction following a given order of candidates.
    Returns the constructed set.
    """
    S = []
    for x in order:
        if _time_remaining() <= 1.0:
            break
        candidate = S + [x]
        if _is_valid(candidate):
            S.append(x)
    return S


def _local_improve(S):
    """
    Try to improve set S by adding elements or swapping.
    Returns an improved set or the original if no improvement found.
    """
    S = list(S)
    improved = True
    while improved and _time_remaining() > 2.0:
        improved = False
        # Try adding each element not in S
        S_set = set(S)
        for x in range(24, 0, -1):
            if x in S_set:
                continue
            if _time_remaining() <= 2.0:
                break
            candidate = S + [x]
            if _is_valid(candidate):
                S = candidate
                S_set = set(S)
                improved = True
                break
        # If no addition worked, try swaps
        if not improved and _time_remaining() > 2.0:
            for i in range(len(S)):
                if _time_remaining() <= 2.0:
                    break
                for x in range(24, 0, -1):
                    if x in S_set:
                        continue
                    candidate = S[:i] + [x] + S[i+1:]
                    if _is_valid(candidate):
                        if len(candidate) >= len(S):  # Only accept if not smaller
                            S = candidate
                            S_set = set(S)
                            improved = True
                            break
                if improved:
                    break
    return S


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    Uses multiple greedy passes with different orderings, then local improvement.
    """
    if _time_remaining() <= 0:
        return list(range(13, 25))
    
    best_S = []
    
    # Try multiple greedy orderings
    orderings = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
        # Try ordering by "safety" - elements with larger reciprocals first (smaller numbers)
        sorted(range(1, 25), key=lambda x: -_R[x]),
        # Try a mixed order: large numbers first, then small
        list(range(24, 12, -1)) + list(range(12, 0, -1)),
    ]
    
    for order in orderings:
        if _time_remaining() <= 2.0:
            break
        S = _greedy_construct(order)
        if len(S) > len(best_S):
            best_S = S
    
    # Local improvement on the best set
    if best_S and _time_remaining() > 3.0:
        best_S = _local_improve(best_S)
    
    if len(best_S) > 0:
        return best_S
    
    # Fallback
    return list(range(13, 25))


if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Cardinality: {len(result)}")
    print(f"Valid: {_is_valid(result)}")
