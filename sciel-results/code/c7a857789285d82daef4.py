"""Rewritten solver for the finite divisor-sum instance (n=5040, target=4979)."""
import time
from collections import deque

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
    Uses meet-in-the-middle for efficiency.
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
    
    # For the left half, generate all subsets of sizes 0..mid
    # Store in a dictionary: (size, sum) -> list of indices
    # Since we want to minimize k, and we check k in increasing order,
    # we just need *one* valid combination for the specific k.
    
    # Optimization: We only care about sums <= _TARGET.
    
    left_sums = {}
    # left has size mid (30). 2^30 is too big to enumerate fully if we do all subsets.
    # But we only need subsets of size <= k.
    # If k is small (e.g., 5), max size in left is min(k, mid).
    # Max size in right is min(k, len(right)).
    
    # Actually, for k=5, max size in left is 5.
    # Number of subsets of size <= 5 from 30 elements:
    # C(30,0)+...+C(30,5) = 1 + 30 + 435 + 4060 + 27405 + 142506 = ~174k.
    # This is very manageable.
    
    # Generate left subsets
    # We use recursion or iterative generation.
    # Let's use a helper to generate subsets of a specific size.
    
    def generate_subsets(arr, max_size, target_sum):
        """
        Yields (size, sum, indices) for subsets of arr with size <= max_size and sum <= target_sum.
        """
        n = len(arr)
        # Use itertools.combinations for each size
        from itertools import combinations
        for size in range(0, min(max_size, n) + 1):
            for comb in combinations(range(n), size):
                s = sum(arr[i] for i in comb)
                if s <= target_sum:
                    yield size, s, comb

    # Store left results
    # key: (size, sum) -> list of index tuples
    # To save memory, we only need one representative for each (size, sum) if we just want to find *a* solution?
    # No, because we need to check if the complement exists in the right.
    # But if multiple left subsets have the same (size, sum), any one works as long as the right complement exists.
    # So we can store just one index tuple for each (size, sum).
    
    left_map = {}
    for size, s, idxs in generate_subsets(left, k, _TARGET):
        key = (size, s)
        if key not in left_map:
            left_map[key] = idxs

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
        for size, s, idxs in generate_subsets(right, r_size, _TARGET):
            if size != r_size:
                continue
            needed_l_sum = _TARGET - s
            if needed_l_sum < 0:
                continue
            key = (l_size, needed_l_sum)
            if key in left_map:
                l_idxs = left_map[key]
                # Combine indices
                # Left indices are 0..mid-1, Right indices are 0..len(right)-1
                # Map right indices back to global indices: + mid
                global_idxs = list(l_idxs) + [i + mid for i in idxs]
                # Extract values
                result = [_DIVS_SMALL[i] for i in global_idxs]
                # Verify
                if len(result) == k and sum(result) == _TARGET and len(set(result)) == k:
                    return result
                # In case of collision or error, we could continue, but it should be correct.
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Check k=1
    if _TARGET in _DIVS_SET:
        return [_TARGET]
        
    # Check k=2
    # O(N^2) is fast enough for N=60
    for i in range(len(_DIVS_SMALL)):
        d1 = _DIVS_SMALL[i]
        rem = _TARGET - d1
        if rem in _DIVS_SET and rem != d1:
            # Ensure rem is in _DIVS_SMALL? If rem > _TARGET, it can't be in _DIVS_SMALL.
            # But we iterate d1 from _DIVS_SMALL.
            # If rem is a divisor and distinct, it's a valid pair.
            # We just need to return [d1, rem].
            # But we must ensure rem is a divisor of 5040. _DIVS_SET contains all.
            # Is it possible rem is not in _DIVS_SMALL? Yes, if rem > _TARGET.
            # But d1 <= _TARGET, so rem = _TARGET - d1 >= 0.
            # If rem > _TARGET, then d1 < 0, impossible.
            # So rem <= _TARGET. Thus if rem is in _DIVS_SET, it is in _DIVS_SMALL.
            return [d1, rem]
            
    # Check k=3, 4, 5, ...
    # We know k=5 is possible (greedy solution).
    # Let's try k=3, 4, 5 using meet-in-the-middle.
    
    for k in range(3, 10):
        if time.time() > deadline:
            break
        sol = _solve_k(k)
        if sol is not None:
            return sol
            
    # Fallback: Greedy solution for k=5
    # 2520 + 1680 + 720 + 56 + 3 = 4979
    # This is a valid solution.
    return [2520, 1680, 720, 56, 3]
