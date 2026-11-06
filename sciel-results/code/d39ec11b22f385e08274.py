import math
import time

def construct():
    N = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    # 1. Get all divisors of N
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            if i > 12:
                divisors.append(i)
            other = N // i
            if other != i and other > 12:
                divisors.append(other)
        i += 1
    
    divisors.sort()
    
    # 2. Iterate M (potential max factor) in increasing order
    # We can optimize by jumping to the next divisor value
    for M in divisors:
        if time.time() > deadline:
            break
            
        # Candidates are divisors <= M
        # Since divisors is sorted, we can find the index
        # But for small M, the list is short.
        candidates = []
        for d in divisors:
            if d > M:
                break
            candidates.append(d)
            
        # Check if a subset of candidates multiplies to N
        # Use DFS
        n_cand = len(candidates)
        # Precompute suffix products for pruning?
        # suffix_prod[i] = product of candidates[i:]
        # This might overflow or be slow to compute if large, but N is fixed.
        # Actually, just checking if current_prod * (N // current_prod) is possible?
        # Simpler pruning: if current_prod > N, fail.
        # if current_prod * (product of remaining) < N, fail.
        
        # Let's compute suffix products of candidates
        suffix_prod = [1] * (n_cand + 1)
        for i in range(n_cand - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i+1] * candidates[i]
            
        def dfs(idx, current_prod, path):
            if current_prod == N:
                return path[:]
            if idx == n_cand:
                return None
            
            # Pruning 1: Exceeded N
            if current_prod > N:
                return None
                
            # Pruning 2: Even multiplying by all remaining, can't reach N
            if current_prod * suffix_prod[idx] < N:
                return None
                
            # Option 1: Skip candidates[idx]
            res = dfs(idx + 1, current_prod, path)
            if res is not None:
                return res
                
            # Option 2: Take candidates[idx]
            # Check divisibility
            next_prod = current_prod * candidates[idx]
            if next_prod <= N and N % next_prod == 0:
                res = dfs(idx + 1, next_prod, path + [candidates[idx]])
                if res is not None:
                    return res
                    
            return None

        result = dfs(0, 1, [])
        if result is not None:
            # Validate result just in case
            prod = 1
            for r in result:
                prod *= r
            if prod == N and all(r > 12 for r in result) and len(set(result)) == len(result):
                return result
            else:
                # Should not happen, but fallback
                continue
                
    # Fallback: return a valid but possibly suboptimal solution
    # The single factor [N] is valid (N > 12, distinct, count 1)
    return [N]
