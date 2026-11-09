"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    target = factorial(12)  # 479001600
    deadline = time.time() + 85  # 85s safety margin within 90s limit
    
    # Generate all divisors of target that are > 12
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)
    
    all_divs = [d for d in get_divisors(target) if d > 12]
    
    best_max = target  # worst case: single factor
    best_factors = [target]
    
    # DFS: build decomposition of remaining product using distinct divisors from candidates
    # We want to minimize the max factor
    def dfs(remaining, candidates, chosen, max_so_far):
        nonlocal best_max, best_factors
        if time.time() > deadline:
            return
        if remaining == 1:
            if max_so_far < best_max:
                best_max = max_so_far
                best_factors = list(chosen)
            return
        if not candidates:
            return
        # Pruning: if remaining > 1 and we can't form it
        if remaining <= 1:
            return
        
        # Lower bound: even with all remaining candidates, can we reach 'remaining'?
        # Product of all remaining candidates must be >= remaining
        # But that's expensive to compute. Instead, use: if remaining is prime and > 12, 
        # it must be in candidates and be the last factor.
        
        for i, c in enumerate(candidates):
            if time.time() > deadline:
                return
            if c > remaining:
                break  # sorted, so all further are too big
            if remaining % c != 0:
                continue
            new_max = max(max_so_far, c)
            if new_max >= best_max:
                # Can't improve, but might still find equal? We want strictly better
                # Actually we can prune if new_max >= best_max since we want min
                continue
            chosen.append(c)
            dfs(remaining // c, candidates[i+1:], chosen, new_max)
            chosen.pop()
    
    dfs(target, all_divs, [], 0)
    
    return best_factors

# EVOLVE-BLOCK-END
