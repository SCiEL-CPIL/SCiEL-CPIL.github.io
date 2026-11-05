"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    best = [target]  # fallback: single factor
    best_max = target
    
    # Prime factorization: 2^10, 3^5, 5^2, 7^1, 11^1
    # We need to partition these into distinct groups, each product > 12
    # Minimize the maximum product
    
    # Strategy: DFS assigning prime factors to "bins" (factors)
    # Try different numbers of factors k from 12 down to 1
    # For each k, try to find distinct products all > 12 minimizing max
    
    # Precompute: we have exponents (10, 5, 2, 1, 1) for primes (2,3,5,7,11)
    primes = [2, 3, 5, 7, 11]
    max_exp = [10, 5, 2, 1, 1]
    
    def dfs_factor_count(k, used_exps, current_products, idx, start_bin):
        """Try to build k distinct factors > 12 by distributing remaining prime factors."""
        nonlocal best, best_max, start_time
        
        if time.time() - start_time > DEADLINE:
            return False
        
        if idx == len(primes):
            # All primes assigned, check if we have exactly k factors all > 12 and distinct
            if len(current_products) == k:
                if all(p > 12 for p in current_products) and len(set(current_products)) == k:
                    mx = max(current_products)
                    if mx < best_max:
                        best_max = mx
                        best = list(current_products)
                        return True
            return False
        
        p = primes[idx]
        exp = used_exps[idx]
        
        # Try assigning 0..exp of prime p to each existing bin, and also try creating new bins
        # This is complex; let's use a different approach
        
        # Actually, let's use a recursive approach where we build factors one at a time
        # Each factor is a divisor of target, > 12, and we need them to multiply to target
        # and be distinct
        
        return False
    
    # Better approach: generate all divisors of target that are > 12
    # Then search for subsets that multiply to target with distinct elements
    
    # Get all divisors
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)
    
    # Actually, let's use a smarter DFS:
    # We build factors one at a time. At each step, we pick a divisor of the remaining product
    # that is > 12, not used before, and we try to keep factors balanced.
    
    def search(remaining, used_set, current_factors, max_so_far):
        nonlocal best, best_max, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if len(current_factors) >= 12:
            if remaining == 1:
                if all(f > 12 for f in current_factors) and len(set(current_factors)) == len(current_factors):
                    mx = max(current_factors)
                    if mx < best_max:
                        best_max = mx
                        best = list(current_factors)
            return
        
        if remaining == 1:
            if len(current_factors) >= 1:
                if all(f > 12 for f in current_factors) and len(set(current_factors)) == len(current_factors):
                    mx = max(current_factors)
                    if mx < best_max:
                        best_max = mx
                        best = list(current_factors)
            return
        
        # Pruning: if max_so_far >= best_max, no improvement possible
        if max_so_far >= best_max:
            return
        
        # Get divisors of remaining that are > 12
        # To keep it manageable, only consider divisors up to best_max
        # and try to pick divisors that keep balance
        
        # Find all divisors of remaining
        divs = []
        for i in range(1, int(remaining**0.5) + 1):
            if remaining % i == 0:
                if i > 12 and i not in used_set and i < best_max:
                    divs.append(i)
                other = remaining // i
                if other != i and other > 12 and other not in used_set and other < best_max:
                    divs.append(other)
        
        # Also consider using remaining itself as a factor (if > 12 and < best_max)
        if remaining > 12 and remaining not in used_set and remaining < best_max:
            divs.append(remaining)
        
        # Sort divisors to try balanced ones first
        # We want to try divisors that keep the product balanced
        # Try divisors close to (remaining)^(1/(max_factors - len(current_factors) + 1))
        max_remaining_factors = 12 - len(current_factors)
        if max_remaining_factors > 1:
            target_size = remaining ** (1.0 / max_remaining_factors)
            divs.sort(key=lambda d: abs(d - target_size))
        else:
            divs.sort()
        
        # Limit the number of divisors we try to avoid explosion
        # Try a subset: the most balanced ones
        max_divs_to_try = 200
        divs = divs[:max_divs_to_try]
        
        for d in divs:
            if d in used_set:
                continue
            if d <= 12:
                continue
            new_max = max(max_so_far, d)
            if new_max >= best_max:
                continue
            
            new_used = used_set | {d}
            new_factors = current_factors + [d]
            search(remaining // d, new_used, new_factors, new_max)
            
            if time.time() - start_time > DEADLINE:
                return
    
    # Start search
    search(target, set(), [], 0)
    
    return best
# EVOLVE-BLOCK-END
