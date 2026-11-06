from math import factorial
from itertools import combinations

def construct():
    target = factorial(12)
    
    # Get all divisors of target
    divisors = []
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other != i:
                divisors.append(other)
    # Filter divisors > 12
    divisors = [d for d in divisors if d > 12]
    # Add 12! itself? No, we want to minimize max, so 12! is a bad start.
    # But we need to consider that the factors must be distinct.
    
    # We search for the smallest M such that a valid decomposition exists with max factor <= M
    # M must be at least 13.
    # Upper bound: 12! (single factor)
    
    best_solution = [target]
    best_max = target
    
    # We can try M from 13 up to best_max
    # To be efficient, we can jump or use binary search, but linear is fine if check is fast.
    # However, check is subset product, which is expensive.
    # Let's try a few specific M values or use a heuristic search first.
    
    # Heuristic: Try to find a solution with max < 100
    # Let's just run a DFS for a specific M, say 50, 100, etc.
    # Or, better, run a general DFS that minimizes max.
    
    def dfs(remaining, last_factor, current_factors, current_max):
        nonlocal best_solution, best_max
        if remaining == 1:
            if current_max < best_max:
                best_max = current_max
                best_solution = current_factors[:]
            return True
        
        # Pruning: if current_max >= best_max, no need to continue
        if current_max >= best_max:
            return False
            
        # Determine candidate factors
        # Must be > last_factor (to ensure distinct and sorted)
        # Must be > 12
        # Must divide remaining
        # Must be <= best_max (to improve)
        
        # We can iterate through divisors of remaining
        # But we need to ensure they are > last_factor
        # To keep it fast, we can just check all divisors of remaining
        
        found = False
        # Get divisors of remaining
        # Optimization: only consider divisors up to best_max
        limit = min(best_max, remaining)
        for i in range(last_factor + 1, int(limit**0.5) + 1):
            if remaining % i == 0:
                # i is a candidate
                if i > 12:
                    dfs(remaining // i, i, current_factors + [i], max(current_max, i))
                    if best_max <= current_max: # Prune if we found a better solution
                        return True
                other = remaining // i
                if other != i:
                    if other > 12 and other <= best_max:
                        dfs(other, other, current_factors + [other], max(current_max, other))
                        if best_max <= current_max:
                            return True
        # Also check if remaining itself is a candidate
        if remaining > last_factor and remaining > 12 and remaining <= best_max:
            dfs(1, remaining, current_factors + [remaining], max(current_max, remaining))
            
        return found

    # Initialize with a single factor solution
    dfs(target, 0, [], 0)
    
    return best_solution
