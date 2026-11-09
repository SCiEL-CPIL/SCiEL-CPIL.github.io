import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    target = math.factorial(12)
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    # Generate all divisors > 12
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if d > 12 and d < target:
                            divisors.append(d)
                        elif d == target:
                            divisors.append(d)
    
    divisors.sort()
    
    # Precompute exponent vectors for each divisor
    cand_info = []
    for d in divisors:
        rem = d
        e_vec = []
        for p in primes:
            c = 0
            while rem % p == 0:
                rem //= p
                c += 1
            e_vec.append(c)
        cand_info.append((d, tuple(e_vec)))
        
    n_cand = len(cand_info)
    
    # Precompute suffix products for pruning
    # suffix_prod[i] = product of all divisors from index i to n_cand-1
    # But we need to be careful: we can only use each divisor once
    # Instead, let's precompute the maximum possible product we can achieve
    # with any subset of divisors from index i onwards
    
    # For better pruning, precompute for each index i the set of exponent vectors
    # that can be formed by any subset of divisors[i:]
    # This is expensive, so let's use a simpler heuristic:
    # Precompute the total exponent contribution available from i onwards
    
    suffix_exps = [[0]*5 for _ in range(n_cand + 1)]
    for i in range(n_cand - 1, -1, -1):
        for j in range(5):
            suffix_exps[i][j] = suffix_exps[i+1][j] + cand_info[i][1][j]
    
    # Find decomposition using DFS with better pruning
    def find_decomposition(max_val):
        # Filter candidates <= max_val
        active_cands = []
        for i in range(n_cand):
            if cand_info[i][0] <= max_val:
                active_cands.append(cand_info[i])
            else:
                break
        
        n_active = len(active_cands)
        if n_active == 0:
            return None
        
        # Precompute suffix exponents for active candidates
        active_suffix_exps = [[0]*5 for _ in range(n_active + 1)]
        for i in range(n_active - 1, -1, -1):
            for j in range(5):
                active_suffix_exps[i][j] = active_suffix_exps[i+1][j] + active_cands[i][1][j]
        
        # DFS with memoization
        # State: (idx, e2, e3, e5, e7, e11, count)
        # Returns list of factors if solution found, else None
        
        memo = {}
        
        def dfs(idx, e2, e3, e5, e7, e11, count, path):
            if time.time() > deadline:
                return None
            
            # Base case: all candidates processed
            if idx == n_active:
                if e2 == 0 and e3 == 0 and e5 == 0 and e7 == 0 and e11 == 0 and count > 0:
                    return list(path)
                return None
            
            # Pruning: if remaining exponents can't be satisfied
            if (e2 > active_suffix_exps[idx][0] or 
                e3 > active_suffix_exps[idx][1] or 
                e5 > active_suffix_exps[idx][2] or 
                e7 > active_suffix_exps[idx][3] or 
                e11 > active_suffix_exps[idx][4]):
                return None
            
            # Pruning: if we've already used 12 factors
            if count > 12:
                return None
            
            # If remaining product is 1, we can't add more factors
            if e2 == 0 and e3 == 0 and e5 == 0 and e7 == 0 and e11 == 0:
                if count > 0:
                    return list(path)
                return None
            
            # Memoization key
            state = (idx, e2, e3, e5, e7, e11, count)
            if state in memo:
                result = memo[state]
                if result is not None:
                    return list(path) + result
                return None
            
            val, vec = active_cands[idx]
            
            # Option 1: Try including active_cands[idx]
            if (vec[0] <= e2 and vec[1] <= e3 and vec[2] <= e5 and 
                vec[3] <= e7 and vec[4] <= e11):
                path.append(val)
                result = dfs(idx + 1, e2 - vec[0], e3 - vec[1], e5 - vec[2], 
                            e7 - vec[3], e11 - vec[4], count + 1, path)
                path.pop()
                
                if result is not None:
                    memo[state] = result
                    return result
            
            # Option 2: Try excluding active_cands[idx]
            result = dfs(idx + 1, e2, e3, e5, e7, e11, count, path)
            
            memo[state] = result
            return result
        
        return dfs(0, 10, 5, 2, 1, 1, 0, [])
    
    # Binary search on M
    lo = 0
    hi = n_cand - 1
    best_solution = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        M = divisors[mid]
        
        sol = find_decomposition(M)
        
        if sol is not None:
            best_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_solution is None:
        return [target]
        
    return best_solution
