import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    target = math.factorial(12)  # 479001600
    
    # Get all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter: must be > 12
    candidates = sorted([d for d in divisors if d > 12])
    n_cands = len(candidates)
    
    # Precompute: for each candidate index, the value
    # We'll use binary search on M
    
    def can_decompose(rem, start_idx, max_val, path, time_limit, max_factors=12):
        """Try to find distinct factors from candidates[start_idx:], all <= max_val,
           whose product is rem. Returns the list if found, None otherwise."""
        if time.time() > time_limit:
            return None
        
        if rem == 1:
            return path[:]
        
        if len(path) >= max_factors:
            return None
        
        if start_idx >= n_cands:
            return None
        
        if rem < 13:
            return None
            
        if candidates[start_idx] > rem:
            return None
            
        # Pruning: check if rem can be achieved with remaining slots
        remaining_slots = max_factors - len(path)
        if remaining_slots <= 0:
            return None
        # rem <= max_val^remaining_slots (necessary condition)
        # Use log to avoid overflow
        if math.log(rem) > remaining_slots * math.log(max_val) + 1e-9:
            return None
        
        # Also: rem must be >= (smallest available factor)^1 * 1 * 1 ... 
        # Actually the minimum product with 1 factor is candidates[start_idx]
        # With k factors, minimum is product of k smallest available
        # But this is expensive to compute. Skip.
        
        for i in range(start_idx, n_cands):
            f = candidates[i]
            if f > max_val:
                break
            if f > rem:
                break
            if rem % f != 0:
                continue
                
            new_rem = rem // f
            new_path = path + [f]
            result = can_decompose(new_rem, i + 1, max_val, new_path, time_limit, max_factors)
            if result is not None:
                return result
        
        return None
    
    # Binary search for minimum M
    lo, hi = 0, n_cands - 1
    best_solution = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        
        mid = (lo + hi) // 2
        M = candidates[mid]
        
        sol = can_decompose(target, 0, M, [], deadline)
        if sol is not None:
            best_solution = sol
            hi = mid - 1  # Try smaller M
        else:
            lo = mid + 1  # Need larger M
    
    if best_solution is None:
        # Fallback: try larger M values from lo onwards
        for idx in range(lo, n_cands):
            if time.time() > deadline:
                break
            M = candidates[idx]
            sol = can_decompose(target, 0, M, [], deadline)
            if sol is not None:
                best_solution = sol
                break
    
    if best_solution is None:
        return [target]
    
    return best_solution
