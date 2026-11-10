import math
import time

def construct():
    start_time = time.time()
    deadline = 80  # safety margin before 90s kill
    
    N = math.factorial(12)  # 479001600
    
    # Global best
    best_max = N
    best_factors = [N]
    
    # Precompute all divisors of N that are > 12
    # N = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Generate all divisors
    divisors = [1]
    primes_exp = [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]
    for p, e in primes_exp:
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    
    # Filter: > 12
    valid_divisors = sorted([d for d in divisors if d > 12])
    valid_set = set(valid_divisors)
    
    # For each valid divisor, precompute its value
    # Build a mapping from divisor to list of divisors of N//divisor that are > 12
    # Actually, let's compute on the fly with caching
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def divisors_of(m, min_val):
        """Return sorted list of divisors of m that are >= min_val."""
        if m < min_val:
            return ()
        result = []
        isqrt_m = int(math.isqrt(m))
        for i in range(min_val, isqrt_m + 1):
            if m % i == 0:
                result.append(i)
                other = m // i
                if other >= min_val and other != i:
                    result.append(other)
        if m >= min_val:
            result.append(m)
        return tuple(sorted(set(result)))
    
    def dfs(remaining_target, last_factor, depth, target_k, path):
        nonlocal best_max, best_factors
        
        if time.time() - start_time > deadline:
            return False
        
        rem = target_k - depth
        
        if rem == 0:
            if remaining_target == 1:
                max_f = path[-1] if path else 0
                if max_f < best_max:
                    best_max = max_f
                    best_factors = list(path)
            return True
        
        if rem == 1:
            x = remaining_target
            if x > last_factor and x > 12:
                if x < best_max:
                    best_max = x
                    best_factors = path + [x]
            return True
        
        # Lower bound: minimum product of 'rem' distinct integers > last_factor
        # (last_factor+1) * (last_factor+2) * ... * (last_factor+rem)
        min_prod = 1
        for i in range(1, rem + 1):
            min_prod *= (last_factor + i)
            if min_prod > remaining_target:
                break
        if remaining_target < min_prod:
            return False
        
        # Upper bound for next factor x:
        # x * (x+1) * ... * (x + rem - 1) <= remaining_target
        # Also x > last_factor, x > 12
        lower_bound = max(last_factor + 1, 13)
        
        # Find upper bound for x
        x_upper = lower_bound
        while True:
            prod_test = 1
            for i in range(rem):
                prod_test *= (x_upper + i)
                if prod_test > remaining_target:
                    break
            if prod_test > remaining_target:
                break
            x_upper += 1
            if x_upper > 50000:
                break
        x_upper = min(x_upper - 1, remaining_target)
        
        # Also bound by best_max - 1
        if best_max < N:
            x_upper = min(x_upper, best_max - 1)
        
        if x_upper < lower_bound:
            return True
        
        # Get divisors of remaining_target in range [lower_bound, x_upper]
        # Use the cached function
        all_divs = divisors_of(remaining_target, lower_bound)
        
        for x in all_divs:
            if time.time() - start_time > deadline:
                return False
            
            if x > x_upper:
                break
            
            if x <= last_factor:
                continue
            
            # Check: remaining_target // x must be achievable by (rem-1) distinct factors > x
            rem_target = remaining_target // x
            
            # Min product of (rem-1) distinct integers > x:
            min_rem = 1
            for i in range(1, rem):
                min_rem *= (x + i)
                if min_rem > rem_target:
                    break
            if rem_target < min_rem:
                continue
            
            # Max product if we want to beat best_max:
            # All remaining factors must be < best_max
            if best_max < N:
                max_rem = 1
                for i in range(rem - 1):
                    max_rem *= (best_max - 1 - i)
                    if max_rem > rem_target:
                        break
                if rem_target > max_rem:
                    continue
            
            path.append(x)
            dfs(rem_target, x, depth + 1, target_k, path)
            path.pop()
            
            if best_max <= 15:
                return True
        
        return True
    
    # Try k=7 first (best geometric mean ~14.48), then k=8, k=6, etc.
    # For k=7: geometric mean = (479001600)^(1/7) ≈ 14.48
    # For k=8: geometric mean ≈ 12.7 (tight, factors must be >12)
    # For k=6: geometric mean ≈ 28
    
    for k in [7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 1]:
        if time.time() - start_time > deadline:
            break
        dfs(N, 12, 0, k, [])
        if best_max <= 15:
            break
    
    return best_factors
