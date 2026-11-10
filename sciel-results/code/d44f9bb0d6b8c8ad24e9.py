import time
import math
from math import factorial
from functools import lru_cache

def construct():
    deadline = time.time() + 55.0
    N = factorial(12)  # 479001600

    # divisors of N = 2^10 * 3^5 * 5^2 * 7 * 11
    primes_exp = [(2,10),(3,5),(5,2),(7,1),(11,1)]
    divisors = [1]
    for p, e in primes_exp:
        new = []
        pw = 1
        for _ in range(e+1):
            for d in divisors:
                new.append(d*pw)
            pw *= p
    divisors = sorted(set(d for d in divisors if d > 12))
    
    # Binary search for the minimum M such that N can be decomposed
    # into distinct factors > 12 and <= M, with at most 12 factors.
    
    # Lower bound: 12th root of N
    lb = int(math.ceil(N ** (1.0/12)))
    if lb < 13: lb = 13
    ub = N
    
    # We need to return the list of factors, not just M.
    # So we find the optimal M, then reconstruct the solution.
    
    def check(M):
        candidates = [d for d in divisors if d <= M]
        if not candidates:
            return None
        
        # Memoization key: (index, remaining_product, count_used)
        # We use a dictionary for memoization to allow clearing between M checks if needed,
        # but since M changes, the candidate list changes, so we can't easily reuse cache across M.
        # However, the number of states is small.
        
        @lru_cache(maxsize=None)
        def dfs(i, rem, count):
            if rem == 1:
                return True if count <= 12 else False
            if count > 12:
                return False
            
            # Pruning: 
            # 1. If we have picked 'count' factors, we need to pick '12-count' more.
            #    The product of the smallest possible remaining factors must be <= rem.
            #    The smallest factors are candidates[i], candidates[i+1], ...
            #    This is hard to compute quickly without precomputation.
            #    Alternative: If rem is large, we need large factors.
            
            # 2. If the product of the largest possible remaining factors is < rem, fail.
            #    Max factors are at the end of candidates.
            #    But we are constrained by 'i' (must be distinct and > previous).
            
            # Let's stick to simple pruning:
            # If rem > 1 and i == len(candidates): return False
            # If rem is prime and rem > candidates[-1] (if i==0) or generally if rem is not in candidates[i:]:
            #    We can check if rem is in candidates[i:]. If yes, we must pick it? No, we can pick factors that multiply to rem.
            
            # Efficient pruning:
            # If rem is not divisible by any candidate in candidates[i:], return False.
            # We can precompute divisors of rem that are in candidates[i:].
            
            for j in range(i, len(candidates)):
                d = candidates[j]
                if d > rem:
                    break
                if rem % d == 0:
                    if dfs(j+1, rem//d, count+1):
                        return True
            return False
        
        if dfs(0, N, 0):
            # Reconstruct solution
            path = []
            i = 0
            rem = N
            count = 0
            while rem != 1:
                for j in range(i, len(candidates)):
                    d = candidates[j]
                    if d > rem:
                        break
                    if rem % d == 0:
                        # Check if this path leads to a solution
                        # We need a function that returns the path or checks existence
                        # To avoid double work, we can modify dfs to return the path or use a separate reconstruction
                        # For now, let's assume we can reconstruct by re-running similar logic
                        # But lru_cache doesn't store the choice.
                        # We can modify dfs to return the list of factors or None.
                        pass
                # This reconstruction is tricky with the current dfs.
                # Let's change dfs to return the list of factors.
                pass
            return None
        return None
    
    # Better approach: DFS that returns the path
    def solve(M):
        candidates = [d for d in divisors if d <= M]
        
        @lru_cache(maxsize=None)
        def dfs(i, rem, count):
            if rem == 1:
                return [] if count <= 12 else None
            if count > 12:
                return None
            
            for j in range(i, len(candidates)):
                d = candidates[j]
                if d > rem:
                    break
                if rem % d == 0:
                    sub = dfs(j+1, rem//d, count+1)
                    if sub is not None:
                        return [d] + sub
            return None
        
        res = dfs(0, N, 0)
        return res

    # Binary search
    best_sol = None
    # Start from a reasonable estimate to speed up
    # Try M = 100, 200, etc? Or just binary search.
    lo = lb
    hi = ub
    
    # To speed up, we can check if a solution exists for a specific M
    # and binary search for the smallest M.
    
    # Optimization: The number of divisors is small.
    # We can iterate M from lb upwards? No, range is large.
    # Binary search is better.
    
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        sol = solve(mid)
        if sol is not None:
            best_sol = sol
            hi = mid
        else:
            lo = mid + 1
            
    if best_sol is None:
        # Fallback to greedy or single factor if binary search failed or timed out
        # Try a few larger M values just in case
        for m in range(100, 500):
             sol = solve(m)
             if sol is not None:
                 best_sol = sol
                 break
                 
    if best_sol is None:
        return [N]
        
    return best_sol
