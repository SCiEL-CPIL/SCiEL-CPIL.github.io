import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    target = math.factorial(12)
    
    # Prime factorization of 12!
    # 12! = 479001600
    # v2(12!) = 10
    # v3(12!) = 5
    # v5(12!) = 2
    # v7(12!) = 1
    # v11(12!) = 1
    target_exp = (10, 5, 2, 1, 1)
    
    # Generate all divisors of 12!
    divisors = []
    sqrt_n = int(math.isqrt(target))
    for i in range(1, sqrt_n + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter and sort
    candidates = [d for d in divisors if d > 12]
    candidates.sort()
    
    # Precompute prime exponent vectors for each candidate
    # primes = [2, 3, 5, 7, 11]
    def get_exp(n):
        exps = [0, 0, 0, 0, 0]
        for i, p in enumerate([2, 3, 5, 7, 11]):
            while n % p == 0:
                exps[i] += 1
                n //= p
        return tuple(exps)
    
    cand_exps = [get_exp(d) for d in candidates]
    
    # We want to find a subset of indices such that sum of exps = target_exp
    # And the max value is minimized.
    # We iterate M = candidates[i] and check if a solution exists using candidates[0..i]
    
    def exists_solution(limit_idx):
        # DFS over indices 0..limit_idx
        # State: (idx, e2, e3, e5, e7, e11, count, path)
        # To avoid copying path, we can just return True/False and store path globally?
        # Or return the path.
        
        # Using recursion with lru_cache is tricky because 'path' is not hashable easily in a way that helps.
        # We just need to know IF a solution exists. Then we can reconstruct or just return one.
        # Let's return the list of indices if found, else None.
        
        # Optimization: Sort candidates by value? They are already sorted.
        # But for pruning, it might be better to consider candidates with larger prime factors first?
        # Let's stick to index order for now, but we can try a heuristic.
        
        # Since we want to minimize the MAX factor, and we are checking a fixed limit M,
        # any subset of {candidates[0..limit_idx]} is valid.
        
        # We can use a set of visited states to avoid recomputation?
        # State: (idx, e2, e3, e5, e7, e11)
        # The number of states is small: (limit_idx+1) * 11 * 6 * 3 * 2 * 2 ~ 792 * 792 ~ 600k max?
        # Actually, the exponents are bounded by target_exp.
        # So the number of states is (limit_idx+1) * (11*6*3*2*2) = (limit_idx+1) * 792.
        # This is very small! We can use memoization.
        
        memo = {}
        
        def dfs(idx, e2, e3, e5, e7, e11):
            if time.time() > deadline:
                return None
            
            if e2 == target_exp[0] and e3 == target_exp[1] and e5 == target_exp[2] and e7 == target_exp[3] and e11 == target_exp[4]:
                return []
            
            if idx > limit_idx:
                return None
            
            # Pruning: if current exps exceed target, invalid (should not happen if we check before adding)
            if e2 > target_exp[0] or e3 > target_exp[1] or e5 > target_exp[2] or e7 > target_exp[3] or e11 > target_exp[4]:
                return None
                
            # Check if we can reach the target with the remaining candidates
            # Simple bound: sum of remaining exponents must be >= required exponents
            # This is hard to compute quickly without precomputation.
            # But the state space is small, so we just recurse.
            
            state = (idx, e2, e3, e5, e7, e11)
            if state in memo:
                return memo[state]
            
            # Option 1: Skip current candidate
            res = dfs(idx + 1, e2, e3, e5, e7, e11)
            if res is not None:
                memo[state] = res
                return res
            
            # Option 2: Take current candidate
            exps = cand_exps[idx]
            ne2 = e2 + exps[0]
            ne3 = e3 + exps[1]
            ne5 = e5 + exps[2]
            ne7 = e7 + exps[3]
            ne11 = e11 + exps[4]
            
            if ne2 <= target_exp[0] and ne3 <= target_exp[1] and ne5 <= target_exp[2] and ne7 <= target_exp[3] and ne11 <= target_exp[4]:
                res = dfs(idx + 1, ne2, ne3, ne5, ne7, ne11)
                if res is not None:
                    # res is a list of indices from the deeper call
                    # We need to add the current index to the path
                    # But wait, the deeper call returns a list of indices.
                    # If we return [idx] + res, that works.
                    memo[state] = [idx] + res
                    return [idx] + res
            
            memo[state] = None
            return None
        
        res = dfs(0, 0, 0, 0, 0, 0)
        if res is None:
            return None
        # Convert indices to values
        return [candidates[i] for i in res]

    # Iterate through possible max factors
    for i, M in enumerate(candidates):
        if time.time() > deadline:
            break
            
        sol = exists_solution(i)
        if sol is not None:
            # Verify distinctness and product (should be correct by construction)
            # Check count <= 12
            if len(sol) <= 12:
                return sol

    # Fallback
    return [target]
