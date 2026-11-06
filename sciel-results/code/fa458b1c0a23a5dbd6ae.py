"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    target = factorial(12)  # 479001600
    
    def check_time():
        return (time.time() - start_time) > DEADLINE
    
    # Generate all divisors of target that are > 12
    divisors = set()
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2**e2) * (3**e3) * (5**e5) * (7**e7) * (11**e11)
                        if d > 12:
                            divisors.add(d)
    
    divisors = sorted(divisors)
    n = len(divisors)
    
    best_max = target
    best_factors = [target]
    
    # Precompute suffix products for pruning
    suffix_prod = [1] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_prod[i] = suffix_prod[i + 1] * divisors[i]
    
    # DFS: pick divisors in increasing order to ensure distinctness
    def dfs(idx, remaining_prod, current_max, chosen, min_val):
        nonlocal best_max, best_factors
        if check_time():
            return
        
        if current_max >= best_max:
            return
        
        if len(chosen) >= 12:
            if remaining_prod == 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(chosen)
            return
        
        if idx >= n:
            return
        
        # Prune: if suffix_prod[idx] < remaining_prod, can't achieve it
        if suffix_prod[idx] < remaining_prod:
            return
        
        # If remaining_prod == 1, we're done
        if remaining_prod == 1:
            if len(chosen) >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(chosen)
            return
        
        # Try each divisor from idx onward
        for i in range(idx, n):
            if check_time():
                return
            d = divisors[i]
            if d < min_val:
                continue
            if d > remaining_prod:
                break
            if remaining_prod % d != 0:
                continue
            new_remaining = remaining_prod // d
            new_max = max(current_max, d)
            if new_max >= best_max:
                continue
            chosen.append(d)
            dfs(i + 1, new_remaining, new_max, chosen, d + 1)
            chosen.pop()
    
    # Start the search
    dfs(0, target, 0, [], 13)
    
    if best_factors is None or len(best_factors) == 0:
        return [target]
    
    return best_factors

# EVOLVE-BLOCK-END
