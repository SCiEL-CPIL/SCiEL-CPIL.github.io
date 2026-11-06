"""Program for large reciprocal-subsum-free sets using LCM-scaled integer arithmetic."""
import time
from math import lcm as _math_lcm

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
    Check if target can be formed by a non-empty subset of elements (integers).
    Uses meet-in-the-middle for larger sets, direct check for small sets.
    """
    if not elements or target <= 0:
        return False

    # Quick check: single element match
    for e in elements:
        if e == target:
            return True

    n = len(elements)
    if n == 0:
        return False

    total = sum(elements)
    if total < target:
        return False

    # For small n, use bitmask enumeration
    if n <= 12:
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
        sums = [0]
        for val in arr:
            new_sums = [s + val for s in sums]
            sums.extend(new_sums)
        return sums

    left_sums = gen_sums(left)
    right_sums = set(gen_sums(right))

    for l_val in left_sums:
        if l_val > target:
            continue
        r_val = target - l_val
        if r_val in right_sums:
            return True

    return False


def _can_add(k, selected):
    """
    Check if we can add k to the selected set S.
    S is a list of integers (the elements themselves, not reciprocals).
    
    When adding k to S to form S' = S ∪ {k}:
    1. Check: Is 1/k a subset sum of reciprocals of S?
    2. For each x in S: Is (1/x - 1/k) a subset sum of reciprocals of S\{x}?
       (Only if 1/x > 1/k, i.e., x < k)
    """
    target_k = _R[k]
    
    # Check 1: Is 1/k a subset sum of reciprocals of S?
    if selected:
        reciprocals_S = [_R[x] for x in selected]
        if _subset_sum_exists(target_k, reciprocals_S):
            return False

    # Check 2: For each x in S where x < k (so 1/x > 1/k)
    for x in selected:
        if x >= k:
            continue  # 1/x <= 1/k, so 1/x - 1/k <= 0, can't be sum of positive terms
        target_x = _R[x] - _R[k]
        if target_x <= 0:
            continue
        remaining = [_R[y] for y in selected if y != x]
        if _subset_sum_exists(target_x, remaining):
            return False

    return True


def _is_valid(S):
    """
    Full validation: Check if set S is reciprocal-subsum-free.
    For each element x in S, 1/x must NOT equal the sum of reciprocals of any subset of S\{x}.
    """
    S = set(S)
    if not S:
        return True
    
    for x in S:
        target = _R[x]
        others = [_R[y] for y in S if y != x]
        if _subset_sum_exists(target, others):
            return False
    return True


def _greedy_build(order):
    """Build a set greedily following the given order."""
    S = []
    for c in order:
        if _time_remaining() < 5.0:
            break
        if _can_add(c, S):
            S.append(c)
    return S


def _local_search_add(S):
    """Try to add any element not in S. Returns (new_set, improved)."""
    S = set(S)
    for c in range(1, 25):
        if c not in S:
            if _can_add(c, list(S)):
                S.add(c)
                return S, True
    return S, False


def _local_search_swap(S):
    """Try swapping one element out and one in, then try to add more. Returns (new_set, improved)."""
    S = set(S)
    size = len(S)
    
    for out_el in list(S):
        if _time_remaining() < 3.0:
            return S, False
        S_minus = S - {out_el}
        for in_el in range(1, 25):
            if in_el in S:
                continue
            # Try adding in_el to S_minus
            if _can_add(in_el, list(S_minus)):
                S_new = S_minus | {in_el}
                # Now try to add more elements
                improved_set, _ = _local_search_add(S_new)
                if len(improved_set) > size:
                    return improved_set, True
    return S, False


def construct():
    """
    Construct the largest subset of [1, 24] that is reciprocal-subsum-free.
    Strategy: Multiple greedy orderings + local search (add + swap).
    """
    if _time_remaining() <= 1.0:
        return list(range(13, 25))

    best_set = set()
    best_size = 0

    # Phase 1: Try several deterministic orderings
    orderings = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
        list(range(12, 0, -1)) + list(range(13, 25)),  # Mid-high then low then high
        list(range(24, 12, -1)) + list(range(1, 13)),  # High then low
        list(range(13, 25)) + list(range(12, 0, -1)),  # High-mid then low-mid
    ]

    for order in orderings:
        if _time_remaining() < 10.0:
            break
        S = _greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Phase 2: Local search on best set - try additions
    improved = True
    while improved and _time_remaining() > 15.0:
        best_set, improved = _local_search_add(best_set)
        if improved:
            best_size = len(best_set)

    # Phase 3: Try swaps to escape local maxima
    if _time_remaining() > 10.0:
        new_set, sw_improved = _local_search_swap(best_set)
        if sw_improved:
            best_set = new_set
            best_size = len(best_set)
            # After a successful swap, try additions again
            improved2 = True
            while improved2 and _time_remaining() > 10.0:
                best_set, improved2 = _local_search_add(best_set)
                if improved2:
                    best_size = len(best_set)

    # Phase 4: Random restarts if time permits
    import random
    random.seed(42)
    
    for trial in range(20):
        if _time_remaining() < 8.0:
            break
        order = list(range(1, 25))
        random.shuffle(order)
        S = _greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            # Local search on this new best
            improved3 = True
            while improved3 and _time_remaining() > 8.0:
                best_set, improved3 = _local_search_add(best_set)
                if improved3:
                    best_size = len(best_set)

    # Final validation
    if not _is_valid(best_set):
        # Fallback to a known valid set
        return list(range(13, 25))

    return list(best_set)


if __name__ == "__main__":
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {sorted(result)}")
    print(f"Valid: {_is_valid(result)}")
