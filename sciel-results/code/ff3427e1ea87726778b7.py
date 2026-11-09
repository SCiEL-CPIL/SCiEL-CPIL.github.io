import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin for 60s limit
    
    target = math.factorial(12)  # 479001600
    
    # Get all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter: must be > 12
    candidates = sorted([d for d in divisors if d > 12])
    n_cand = len(candidates)
    
    # Precompute: for each divisor d of target (d > 12), the list of candidate indices that divide d
    # This allows fast lookup in DFS
    div_to_indices = {}
    for d in divisors:
        if d <= 12:
            continue
        indices = []
        for i, c in enumerate(candidates):
            if c > d:
                break
            if d % c == 0:
                indices.append(i)
        div_to_indices[d] = indices
    
    # Map value to index for quick lookup
    val_to_idx = {c: i for i, c in enumerate(candidates)}
    
    best_solution = None
    
    def dfs(rem, start_idx, max_val, path, time_limit):
        if time.time() > time_limit:
            return None
        
        if rem == 1:
            return path[:]
        
        if len(path) >= 12:
            return None
        
        if rem < 13:
            return None
        
        # Quick check: if rem itself is a valid factor we can use
        if 12 < rem <= max_val:
            idx = val_to_idx.get(rem, -1)
            if idx >= start_idx:
                return path + [rem]
        
        if rem not in div_to_indices:
            return None
        
        for idx in div_to_indices[rem]:
            if idx < start_idx:
                continue
            f = candidates[idx]
            if f > max_val:
                break
            if f > rem:
                break
            
            new_rem = rem // f
            new_path = path + [f]
            result = dfs(new_rem, idx + 1, max_val, new_path, time_limit)
            if result is not None:
                return result
        
        return None
    
    # Iterate M from smallest candidate upward
    for M in candidates:
        if time.time() > deadline:
            break
        
        sol = dfs(target, 0, M, [], deadline)
        if sol is not None:
            best_solution = sol
            break
    
    if best_solution is None:
        return [target]
    
    return best_solution
