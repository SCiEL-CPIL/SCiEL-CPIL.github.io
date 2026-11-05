import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 82.0  # Safety margin before 90s kill
    
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Precompute all divisors of target
    all_divs = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
    all_divs = sorted(all_divs)
    
    # For each divisor, precompute its divisors > 12
    divisors_cache = {}
    for d in all_divs:
        divs_of_d = []
        for j in range(1, int(d**0.5) + 1):
            if d % j == 0:
                if j > 12:
                    divs_of_d.append(j)
                other = d // j
                if other != j and other > 12:
                    divs_of_d.append(other)
        divisors_cache[d] = sorted(divs_of_d)
    
    # Binary search on the maximum factor M
    # Lower bound: at least 13 (smallest integer > 12)
    # Upper bound: target itself (trivial solution)
    
    def feasible(M):
        """Check if we can decompose target into 1-12 distinct integers > 12, all <= M."""
        # DFS with memoization
        # State: (rem, last_factor, count_used)
        # We want to pick factors > last_factor, <= M, distinct, product = rem
        
        memo = set()
        
        def dfs(rem, last, count):
            if time.time() - start_time > deadline:
                return False
            
            if rem == 1:
                return 1 <= count <= 12
            
            if count >= 12:
                # Must use rem as the last factor
                if rem > last and rem <= M and rem > 12:
                    return True
                return False
            
            state = (rem, last, count)
            if state in memo:
                return False
            memo.add(state)
            
            # Get candidate divisors of rem that are > last and <= M
            candidates = divisors_cache.get(rem, [])
            
            for d in candidates:
                if d > last and d <= M:
                    # Pruning: if rem // d < d + 1 and rem // d != 1, 
                    # we can't pick any more factors > d, so rem//d must be 1
                    # Actually, if rem//d > 1, we need at least one more factor > d
                    # So rem//d must have a divisor > d, which means rem//d >= d+1
                    # Or rem//d == 1
                    if rem // d == 1:
                        if dfs(1, d, count + 1):
                            return True
                    else:
                        # Need rem//d >= d + 1 for possibility of another factor > d
                        # Actually this isn't strictly true - we could have more factors
                        # But the minimum next factor is d+1, so rem//d must be >= (d+1)
                        # if we need at least one more factor
                        if rem // d >= d + 1:
                            if dfs(rem // d, d, count + 1):
                                return True
                        elif rem // d > 12:
                            # rem//d could be the last factor if it's > d
                            if rem // d > d and rem // d <= M:
                                if dfs(rem // d, d, count + 1):
                                    return True
                        else:
                            # rem//d > 1 but < d+1, and not a valid last factor
                            # Can't proceed
                            pass
            
            return False
        
        return dfs(target, 12, 0)
    
    # Binary search for minimal M
    lo = 13
    hi = target
    
    # First, verify that hi is feasible (it should be - just [target])
    if not feasible(hi):
        # Should not happen, but return the trivial solution
        return [target]
    
    while lo < hi:
        mid = (lo + hi) // 2
        if time.time() - start_time > deadline:
            break
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    
    best_M = hi
    
    # Now find the actual decomposition with max factor = best_M
    best_path = None
    
    def find_path(rem, last, count, path, M):
        if time.time() - start_time > deadline:
            return None
        
        if rem == 1:
            if 1 <= count <= 12:
                return list(path)
            return None
        
        if count >= 12:
            if rem > last and rem <= M and rem > 12:
                new_path = path + [rem]
                return new_path
            return None
        
        candidates = divisors_cache.get(rem, [])
        
        for d in candidates:
            if d > last and d <= M:
                if rem // d == 1:
                    result = find_path(1, d, count + 1, path + [d], M)
                    if result is not None:
                        return result
                else:
                    if rem // d >= d + 1 or (rem // d > d and rem // d <= M and rem // d > 12):
                        result = find_path(rem // d, d, count + 1, path + [d], M)
                        if result is not None:
                            return result
        
        return None
    
    result = find_path(target, 12, 0, [], best_M)
    
    if result is not None:
        return result
    else:
        # Fallback: return the trivial solution
        return [target]
