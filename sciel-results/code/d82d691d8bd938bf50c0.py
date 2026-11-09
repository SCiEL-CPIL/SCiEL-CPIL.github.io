import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    target = math.factorial(12)
    
    # Precompute all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter divisors > 12 and sort
    candidates = sorted([d for d in divisors if d > 12])
    
    def can_decompose(D):
        """Check if target can be decomposed into distinct factors from candidates <= D."""
        cands = [d for d in candidates if d <= D]
        if not cands:
            return None
        
        cand_set = set(cands)
        solution = None
        
        # Memoization: (rem, min_idx) -> False if we've already determined no solution from this state
        # We only store False to prune. True is returned immediately.
        memo = set()
        
        def dfs(rem, min_idx, path):
            nonlocal solution
            if solution is not None:
                return
            if time.time() > deadline:
                return
            
            if rem == 1:
                solution = path[:]
                return
            
            if len(path) >= 12:
                return
            
            # Pruning: If rem is in cands and its index > min_idx, we can finish immediately
            if rem in cand_set:
                idx = bisect.bisect_left(cands, rem)
                if idx < len(cands) and cands[idx] == rem and idx > min_idx:
                    solution = path + [rem]
                    return
            
            key = (rem, min_idx)
            if key in memo:
                return
            
            # Iterate candidates from min_idx onwards
            for i in range(min_idx, len(cands)):
                d = cands[i]
                if d > rem:
                    break
                if rem % d != 0:
                    continue
                
                dfs(rem // d, i + 1, path + [d])
                if solution is not None:
                    return
            
            memo.add(key)
        
        dfs(target, 0, [])
        return solution

    # Search for minimum D
    # The known optimal solution has max factor 33.
    # Let's try a linear search from 13 upwards first, as the answer is likely small.
    # If not found quickly, fall back to binary search.
    
    best_sol = None
    found_D = None
    
    # Linear search for small D (13 to 100)
    for D in range(13, 101):
        if time.time() > deadline:
            break
        sol = can_decompose(D)
        if sol is not None:
            best_sol = sol
            found_D = D
            break
    
    # If not found in linear search, try binary search on [101, target]
    if best_sol is None:
        low = 101
        high = target
        while low < high:
            if time.time() > deadline:
                break
            mid = (low + high) // 2
            sol = can_decompose(mid)
            if sol is not None:
                high = mid
                best_sol = sol
                found_D = mid
            else:
                low = mid + 1
        
        # Check the final low value if loop ended without finding (or just to be sure)
        if best_sol is None and low <= high:
            sol = can_decompose(low)
            if sol is not None:
                best_sol = sol
                found_D = low
    
    if best_sol:
        return best_sol
    else:
        return [target]
