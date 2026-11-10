import math
import time

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Precompute all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.sort()
    
    # Build a set for O(1) lookup
    divisor_set = set(divisors)
    
    # For each divisor, precompute which divisors are greater than it
    # This is expensive, so we'll compute on the fly with caching
    
    # Global best
    best_max_val = N
    best_factors = [N]
    
    # Precompute min_prod[last_factor][remaining] for quick pruning
    # last_factor ranges from 12 to maybe 200
    # remaining ranges from 1 to 7
    max_lf = 300
    max_rem = 8
    min_prod_table = [[1] * (max_rem + 1) for _ in range(max_lf + 1)]
    for lf in range(13, max_lf + 1):
        min_prod_table[lf][0] = 1
        for r in range(1, max_rem + 1):
            min_prod_table[lf][r] = min_prod_table[lf][r-1] * (lf + r)
    
    def get_min_prod(last_factor, remaining):
        if last_factor >= max_lf or remaining >= max_rem:
            # Compute on the fly
            prod = 1
            for i in range(1, remaining + 1):
                prod *= (last_factor + i)
            return prod
        return min_prod_table[last_factor][remaining]
    
    def dfs(target_k, current_prod, last_factor, depth, path):
        nonlocal best_max_val, best_factors
        
        if time.time() - start_time > 55:
            return
        
        if depth == target_k:
            if current_prod == N:
                max_f = max(path)
                if max_f < best_max_val:
                    best_max_val = max_f
                    best_factors = list(path)
            return
        
        remaining = target_k - depth
        
        if N % current_prod != 0:
            return
        target_rem = N // current_prod
        
        # Pruning: minimum possible product of remaining factors
        min_possible = get_min_prod(last_factor, remaining)
        if target_rem < min_possible:
            return
        
        # Also, the maximum factor in our path so far is last_factor (since we build in increasing order)
        # The final max will be at least the last element we add
        # If even the smallest possible completion gives a max >= best_max_val, we might still want to continue
        # but we can prune if the minimum possible max is already >= best_max_val
        # Actually, since we're adding factors in increasing order, the last factor added will be the max
        # So if we can guarantee that any completion will have max >= best_max_val, we can prune
        # The minimum possible value for the next factor is last_factor + 1
        # The minimum possible value for the last factor (the max) is at least last_factor + remaining
        # So if last_factor + remaining >= best_max_val, we can potentially prune
        # But this is a weak bound. Let's use it carefully.
        
        if remaining == 1:
            x = target_rem
            if x > last_factor and x > 12 and x < best_max_val:
                path.append(x)
                dfs(target_k, current_prod * x, x, depth + 1, path)
                path.pop()
            return
        
        lower_bound = max(last_factor + 1, 13)
        
        # Upper bound: x * (x+1) * ... * (x+remaining-1) <= target_rem
        # x^remaining <= target_rem roughly
        x_upper = int(target_rem ** (1.0 / remaining)) + 10
        # But also x must be less than best_max_val (since we want to improve)
        x_upper = min(x_upper, best_max_val - 1)
        
        if x_upper < lower_bound:
            return
        
        # Iterate through divisors of target_rem in range [lower_bound, x_upper]
        # Instead of iterating all integers, iterate through divisors
        for x in divisors:
            if x < lower_bound:
                continue
            if x > x_upper:
                break
            
            if target_rem % x != 0:
                continue
            
            # Additional pruning: after choosing x, the remaining product is target_rem // x
            # The remaining (remaining-1) factors must be > x
            # Min product of those is get_min_prod(x, remaining - 1)
            rem_after = target_rem // x
            if rem_after < get_min_prod(x, remaining - 1):
                continue
            
            # Also, the max factor in this branch will be at least x (and possibly larger)
            # Since we're building in increasing order, the last factor will be the max
            # We can't easily bound it without going deeper, so let's recurse
            
            path.append(x)
            dfs(target_k, current_prod * x, x, depth + 1, path)
            path.pop()
    
    # Search from largest k down to smallest
    # More factors generally means smaller max factor
    for k in range(7, 0, -1):
        if time.time() - start_time > 55:
            break
        dfs(k, 1, 12, 0, [])
    
    return best_factors
