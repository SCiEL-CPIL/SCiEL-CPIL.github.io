"""Solve: factor 12! into 1..12 DISTINCT integers > 12, minimize the LARGEST factor."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def _all_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i > 12:
                divs.append(i)
            if i != n // i and n // i > 12:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def construct():
    deadline = time.time() + 80  # safety margin under 90s kill
    target = factorial(12)
    divs = _all_divisors(target)
    n = len(divs)
    divs_set = set(divs)
    
    best_max = target  # single factor is always valid
    best_sol = [target]
    
    # DFS: pick a subset of divs (in increasing order) whose product equals target
    # Pruning:
    # 1. If current factor >= best_max, prune (we want to minimize max)
    # 2. If remaining product < 1, prune
    # 3. If we've picked 12 factors, check if product matches
    # 4. If remaining product is 1, we're done
    # 5. Key pruning: the smallest possible factor we can still pick is divs[idx]
    #    If remaining_product / (product of all remaining available divisors) > 1, prune
    #    More practically: if remaining_product has a prime factor > best_max, prune
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Precompute: for each index, the product of all divs from that index onward
    # (capped at some reasonable value to avoid overflow)
    # Actually, let's use a different approach:
    # At each step, we know remaining = target // current_product
    # We need to pick distinct divs > 12, all >= divs[idx], whose product is remaining
    # If remaining == 1: success
    # If remaining < divs[idx] and remaining > 1: prune (can't form with factors >= divs[idx])
    # If remaining is not an integer: shouldn't happen if we always divide by divisors
    # If any factor we pick >= best_max: prune
    
    # Better pruning: factorize remaining and check if all prime factors <= best_max-1
    # Actually simpler: if remaining > 1 and remaining < divs[idx]: prune
    # Also: if we need more than (12 - count) factors, but the minimum product of 
    # (12-count) factors starting from divs[idx] exceeds remaining... complex
    
    # Let's use a cleaner approach with memoization on (remaining, idx, count)
    # But remaining can be large. Let's just do DFS with good pruning.
    
    def dfs(remaining, idx, count, current_factors):
        nonlocal best_max, best_sol, deadline
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            # Found a valid solution
            max_factor = max(current_factors) if current_factors else 0
            if max_factor < best_max:
                best_max = max_factor
                best_sol = list(current_factors)
            return
        
        if count >= 12:
            return
        
        # Pruning: if remaining < 1, invalid
        if remaining < 1:
            return
        
        # If remaining > 1, we need at least one more factor
        # The smallest available factor is divs[idx]
        # If remaining < divs[idx], we can't form it (all remaining divs are >= divs[idx])
        if remaining < divs[idx]:
            return
        
        # Pruning: if remaining itself is a divisor > 12 and >= divs[idx], 
        # we could take it as the last factor, but only if it's < best_max
        # This is handled by the loop below
        
        # Key pruning: if remaining has a prime factor > best_max - 1, we can't use it
        # (since all factors must be < best_max to improve)
        # Actually, factors must be < best_max to strictly improve.
        # But we also allow equal? No, we want to minimize, so < best_max is needed for improvement.
        # However, we still need to search for solutions with max == best_max? No, we already have one.
        # So all factors in a new solution must be < best_max to be useful.
        
        # Check: can remaining be factored into numbers all < best_max?
        # Quick check: if remaining > 1 and remaining < best_max and remaining > 12 and remaining not in divs_set:
        #   then remaining must be composite, and we need to factor it
        # Simpler: just try the loop and let the recursion handle it
        
        for i in range(idx, n):
            d = divs[i]
            if d >= best_max:
                break  # since divs is sorted, all subsequent are also >= best_max
            
            if remaining % d != 0:
                continue
            
            current_factors.append(d)
            dfs(remaining // d, i + 1, count + 1, current_factors)
            current_factors.pop()
            
            if time.time() > deadline:
                return
        
        # Also try: can we stop here? Only if remaining == 1, which is handled above.
        # But what if we don't pick any more factors and remaining != 1? That's invalid.
    
    dfs(target, 0, 0, [])
    
    return best_sol
# EVOLVE-BLOCK-END
