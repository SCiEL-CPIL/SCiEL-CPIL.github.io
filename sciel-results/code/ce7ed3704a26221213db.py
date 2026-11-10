"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

def construct():
    """
    Find 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    target = math.factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of 12! that are > 12
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = []
    for a in range(11):      # 2^a, a in 0..10
        for b in range(6):   # 3^b, b in 0..5
            for c in range(3):  # 5^c, c in 0..2
                for d in range(2):  # 7^d, d in 0..1
                    for e in range(2):  # 11^e, e in 0..1
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.append(val)
    
    divisors.sort()
    
    # We need to find a subset of distinct elements from divisors
    # whose product is exactly target, minimizing the maximum element.
    
    # Strategy: DFS with strong pruning
    # State: remaining product, minimum allowed factor, set of used factors
    # We want to minimize the max factor.
    
    # Key pruning: 
    # 1. If remaining == 1, we're done
    # 2. If remaining > 12 and remaining is not used, we can take it as the last factor
    # 3. For splitting, try divisors d of remaining where d >= min_factor, d > 12, d not used
    # 4. Prune if new_remaining < next_min_factor (and new_remaining != 1)
    # 5. Prune if current_max >= best_max
    
    best_solution = None
    best_max = float('inf')
    
    # For efficiency, precompute which divisors divide which numbers
    # Actually, let's just do the DFS carefully.
    
    # Important: to minimize the max, we should try to use many small factors.
    # The number of factors can be 1 to 12.
    
    # Let's think about lower bounds on the max factor:
    # If we use k factors, the minimum possible product is 13*14*...*(12+k)
    # 13*14*15*16*17*18*19 = 253955520 < 479001600
    # 13*14*15*16*17*18*19*20 = 5079110400 > 479001600
    # So max 7 factors of consecutive small numbers.
    
    # But we don't have to use consecutive numbers.
    
    # Let's do a DFS that tries to build the factor set.
    # To minimize the max, we'll try to keep factors as small as possible.
    
    # Approach: iterate over possible max values M from small to large.
    # For each M, check if there's a valid decomposition using only factors in (12, M].
    # The first M for which a solution exists is our answer.
    
    # But this might be slow. Let's do a DFS that tracks the current max and prunes.
    
    def dfs(remaining, min_factor, used_set, current_max):
        nonlocal best_solution, best_max, deadline, start_time
        
        # Time check
        if time.time() - start_time > 85:
            return False  # timeout
        
        # Pruning: if current_max >= best_max, no point
        if current_max >= best_max:
            return False
        
        # If remaining is 1, we have a complete solution
        if remaining == 1:
            if len(used_set) >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(used_set)
            return True
        
        # If remaining > 12 and remaining >= min_factor and not used, 
        # we can use it as the final factor
        if remaining > 12 and remaining >= min_factor and remaining not in used_set:
            new_max = max(current_max, remaining)
            if new_max < best_max:
                used_set.add(remaining)
                dfs(1, remaining + 1, used_set, new_max)
                used_set.discard(remaining)
        
        # Try to split remaining into d * (remaining/d)
        # where d >= min_factor, d > 12, d not in used_set
        # and remaining/d >= d+1 (to ensure distinctness and ordering)
        # Actually, remaining/d just needs to be >= min_factor for the next level
        # but since we enforce min_factor = d+1, remaining/d >= d+1
        
        for d in divisors:
            if d < min_factor:
                continue
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            if d in used_set:
                continue
            
            new_remaining = remaining // d
            new_max = max(current_max, d)
            
            # Pruning: new_max >= best_max means no improvement possible
            if new_max >= best_max:
                break  # divisors are sorted, so larger d gives larger or equal new_max
            
            # Pruning: new_remaining must be expressible with factors > d
            # The smallest possible factor for the next level is d+1
            # If new_remaining == 1, handled above
            # If new_remaining > 1, we need new_remaining >= d+1
            # (since all remaining factors must be > d, i.e., >= d+1)
            if new_remaining > 1 and new_remaining < d + 1:
                continue
            
            # Additional pruning: new_remaining must be a product of distinct
            # integers all > d, each dividing 12!
            # Quick check: new_remaining must be >= (d+1) if it's a single factor,
            # or >= (d+1)*(d+2) if two factors, etc.
            # But we don't know how many factors remain.
            
            # Let's also check: if new_remaining is prime and new_remaining < d+1, skip
            # (already covered by new_remaining < d+1 check)
            
            # Pruning based on number of factors:
            # We can have at most 12 - len(used_set) - 1 more factors (since we're adding d)
            # Actually, we can have at most 12 - len(used_set) factors total after adding d
            # Wait, used_set already has the current factors, we're about to add d
            # So after adding d, we'll have len(used_set) + 1 factors
            # We can have at most 12 total, so at most 12 - len(used_set) - 1 more after this
            # But this is a loose bound.
            
            # Stronger pruning: the minimum product of (12 - len(used_set)) distinct factors > d
            # is (d+1)*(d+2)*...*(d + 12 - len(used_set))
            # If new_remaining is less than this, it's impossible.
            # But computing this for each step might be expensive.
            
            # Let's just do the recursion
            used_set.add(d)
            dfs(new_remaining, d + 1, used_set, new_max)
            used_set.discard(d)
            
            # Early termination: if we found a solution with max <= d, 
            # no need to try larger d
            if best_max <= d:
                break
        
        return best_solution is not None
    
    # Start the DFS
    dfs(target, 13, set(), 0)
    
    if best_solution is not None:
        return best_solution
    else:
        # Fallback: return [12!] (single factor)
        return [target]

# EVOLVE-BLOCK-END
