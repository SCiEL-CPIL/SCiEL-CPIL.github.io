"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import factorial

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s kill
    N = factorial(12)  # 479001600
    MIN_FACTOR = 13
    MAX_FACTORS = 12
    
    # Precompute all divisors of N
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
        i += 1
    divisors.sort()
    
    # Filter divisors that are >= MIN_FACTOR
    valid_divs = [d for d in divisors if d >= MIN_FACTOR]
    
    # Prime factorization of N for quick divisibility checks
    # N = 12! = 2^10 * 3^5 * 5^2 * 7^1
    # We'll use a different approach: just check if remaining is divisible
    
    best = None
    best_max = N  # initial best is the single factor N
    
    def can_form_product(remaining, min_val, max_factors_left):
        """Check if remaining can be factored into at most max_factors_left distinct factors all >= min_val."""
        if remaining == 1:
            return True
        if max_factors_left <= 0:
            return False
        if max_factors_left == 1:
            return remaining >= min_val
        # The smallest possible product of max_factors_left distinct integers >= min_val
        # is min_val * (min_val+1) * ... * (min_val + max_factors_left - 1)
        # If remaining is less than that, it's impossible
        min_prod = 1
        for i in range(max_factors_left):
            min_prod *= (min_val + i)
        if remaining < min_prod:
            return False
        # Also, remaining must be divisible by at least one valid divisor >= min_val
        # (this is a weaker check but helps prune)
        return True
    
    def dfs(remaining, start_idx, chosen, current_max):
        nonlocal best, best_max
        if time.time() - t0 > DEADLINE:
            return
        
        if remaining == 1:
            if chosen:
                m = current_max
                if m < best_max:
                    best_max = m
                    best = list(chosen)
            return
        
        if len(chosen) >= MAX_FACTORS:
            return
            
        # If we can't form the product with remaining factors, prune
        factors_left = MAX_FACTORS - len(chosen)
        if not can_form_product(remaining, valid_divs[start_idx] if start_idx < len(valid_divs) else MIN_FACTOR, factors_left):
            return
        
        # Try each possible next factor
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max:
                # Since divisors are sorted, all subsequent d' >= d will give new_max' >= new_max >= best_max
                # So we can break
                break
            
            chosen.append(d)
            dfs(remaining // d, i + 1, chosen, new_max)
            chosen.pop()
            
            if time.time() - t0 > DEADLINE:
                return
    
    dfs(N, 0, [], 0)
    
    if best is None:
        return [N]
    return best
# EVOLVE-BLOCK-END
