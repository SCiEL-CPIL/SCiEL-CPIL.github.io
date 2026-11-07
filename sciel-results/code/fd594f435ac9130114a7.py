import time
import sys
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM of all divisors to determine the period of the pattern
    # LCM(6, 10, 14, 15, 21, 22, 26, 33, 35, 39)
    # 2, 3, 5, 7, 11, 13 -> 30030
    lcm_val = 1
    for d in divisors:
        lcm_val = lcm_val * d // gcd(lcm_val, d)
        
    # We only need to check L in range [1, lcm_val]
    # because the relative positions of multiples repeat every lcm_val
    # Also we need L+W <= max_val, so L <= max_val - W.
    # Since lcm_val is 30030, and W will be small, L will be small.
    
    def check_assignment(L, W):
        """
        Returns a list of chosen values if a valid distinct assignment exists
        in [L, L+W], otherwise None.
        """
        candidates = []
        for d in divisors:
            # Find multiples of d in [L, L+W]
            start_k = (L + d - 1) // d
            end_k = (L + W) // d
            cands = [k * d for k in range(start_k, end_k + 1)]
            if not cands:
                return None
            candidates.append(cands)
            
        # Backtracking to find distinct values
        # Sort candidates by length to prune faster (MRV heuristic)
        # But we need to keep track of original index to return in correct order
        indexed_cands = [(i, cands) for i, cands in enumerate(candidates)]
        indexed_cands.sort(key=lambda x: len(x[1]))
        
        assignment = [None] * n
        used_values = set()
        
        def backtrack(idx):
            if idx == len(indexed_cands):
                return True
            
            orig_idx, cands = indexed_cands[idx]
            for val in cands:
                if val not in used_values:
                    assignment[orig_idx] = val
                    used_values.add(val)
                    if backtrack(idx + 1):
                        return True
                    used_values.remove(val)
                    assignment[orig_idx] = None
            return False
        
        if backtrack(0):
            return assignment
        return None

    def is_feasible(W):
        """
        Checks if there exists an L such that [L, L+W] contains a valid distinct assignment.
        Returns the assignment if found, else None.
        """
        # L must be such that L+W <= max_val
        max_L = max_val - W
        # Also we only care about L in [1, lcm_val] due to periodicity
        # But if max_L < lcm_val, we just check up to max_L.
        # If max_L >= lcm_val, we check 1..lcm_val.
        # Actually, if a solution exists for some L0, it exists for L0 + k*lcm_val.
        # We want the smallest values, so we prefer small L.
        # We iterate L from 1 to min(lcm_val, max_L).
        
        upper_L = min(lcm_val, max_L)
        
        for L in range(1, upper_L + 1):
            res = check_assignment(L, W)
            if res is not None:
                return res
        return None

    # Binary search for minimum W
    # Lower bound: 1 (since distinct integers, min diff is 1, but span is max-min)
    # Actually, if we pick 1, 2, 3... span is n-1. But they must be multiples.
    # Let's start low.
    low = 1
    high = 1000000 # Safe upper bound
    
    best_res = None
    best_W = None
    
    # Time guard
    start_time = time.time()
    timeout = 50.0 # 50s limit to be safe
    
    while low <= high:
        if time.time() - start_time > timeout:
            break
            
        mid = (low + high) // 2
        
        res = is_feasible(mid)
        
        if res is not None:
            best_res = res
            best_W = mid
            high = mid - 1
        else:
            low = mid + 1
            
    # If binary search didn't find anything (unlikely), fallback to a larger search or return divisors
    if best_res is None:
        # Fallback: just return the divisors themselves (valid, distinct, range is large)
        # Or try a linear scan for small W just in case
        for W in range(1, 5000):
            res = is_feasible(W)
            if res is not None:
                best_res = res
                best_W = W
                break
        if best_res is None:
            return divisors[:]

    # Ensure types are Python int
    return [int(x) for x in best_res]
