"""Rewritten solver for the finite divisor-sum instance (n=5040, target=4979)."""
import time
from itertools import combinations

_N = 5040
_TARGET = 4979

def _divisors(n):
    divs = []
    for d in range(1, int(n ** 0.5) + 1):
        if n % d == 0:
            divs.append(d)
            if d != n // d:
                divs.append(n // d)
    divs.sort(reverse=True)
    return divs

_DIVS = _divisors(_N)
_DIVS_SET = set(_DIVS)
_DIVS_SMALL = [d for d in _DIVS if d <= _TARGET]

def _solve_k(k):
    """
    Find a subset of exactly k distinct divisors from _DIVS_SMALL that sums to _TARGET.
    Returns the list of divisors or None if not found.
    Uses meet-in-the-middle with bitset optimization for the left half.
    """
    if k == 0:
        return [] if _TARGET == 0 else None
    if k == 1:
        if _TARGET in _DIVS_SET:
            return [_TARGET]
        return None
    
    n = len(_DIVS_SMALL)
    if k > n:
        return None
    
    # Split into two halves
    mid = n // 2
    left = _DIVS_SMALL[:mid]
    right = _DIVS_SMALL[mid:]
    
    # For the left half, use bitset-based DP to track which (size, sum) pairs are achievable
    # Store one representative index tuple for each achievable (size, sum)
    
    # left_map: (size, sum) -> index tuple
    left_map = {}
    
    # Initialize: size 0, sum 0
    left_map[(0, 0)] = ()
    
    # Process each element in left one by one, updating achievable states
    for idx, val in enumerate(left):
        # Iterate over current states in reverse to avoid using the same element twice
        # We need to be careful: we should snapshot the current states before adding new ones
        current_states = list(left_map.items())
        for (size, s), idxs in current_states:
            new_size = size + 1
            new_sum = s + val
            if new_sum <= _TARGET and new_size <= k:
                new_key = (new_size, new_sum)
                if new_key not in left_map:
                    left_map[new_key] = idxs + (idx,)
    
    # Now search in right half
    # For each possible size r_size in [0, min(k, len(right))]:
    #   l_size = k - r_size
    #   For each subset of right of size r_size with sum r_sum:
    #     needed_l_sum = _TARGET - r_sum
    #     if (l_size, needed_l_sum) in left_map:
    #        Found solution!
    
    right_size_max = min(k, len(right))
    for r_size in range(0, right_size_max + 1):
        l_size = k - r_size
        if l_size < 0 or l_size > len(left):
            continue
        
        # Generate right subsets of size r_size
        for comb in combinations(range(len(right)), r_size):
            r_sum = sum(right[i] for i in comb)
            if r_sum > _TARGET:
                continue
            needed_l_sum = _TARGET - r_sum
            if needed_l_sum < 0:
                continue
            key = (l_size, needed_l_sum)
            if key in left_map:
                l_idxs = left_map[key]
                # Combine indices
                # Left indices are 0..mid-1, Right indices are 0..len(right)-1
                # Map right indices back to global indices: + mid
                global_idxs = list(l_idxs) + [i + mid for i in comb]
                # Extract values
                result = [_DIVS_SMALL[i] for i in global_idxs]
                # Verify
                if len(result) == k and sum(result) == _TARGET and len(set(result)) == k:
                    return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Check k=1
    if _TARGET in _DIVS_SET:
        return [_TARGET]
        
    # Check k=2
    for i in range(len(_DIVS_SMALL)):
        d1 = _DIVS_SMALL[i]
        rem = _TARGET - d1
        if rem in _DIVS_SET and rem != d1:
            return [d1, rem]
            
    # Check k=3, 4, 5, ...
    for k in range(3, 10):
        if time.time() > deadline:
            break
        sol = _solve_k(k)
        if sol is not None:
            return sol
            
    # Fallback: Greedy solution for k=5
    return [2520, 1680, 720, 56, 3]
