import math
from functools import lru_cache

def construct():
    N = 12
    target = math.factorial(N)
    
    # 1. Generate all divisors of target
    divisors = []
    # Efficient way to get divisors
    i = 1
    while i * i <= target:
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
        i += 1
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    # Set for O(1) lookup? Not strictly necessary if we iterate and check modulo, 
    # but checking if d divides rem_prod is fast.
    
    best_max = target
    best_solution = [target]
    
    # To speed up, we can pre-compute which divisors divide which? 
    # No, rem_prod changes.
    
    # We will iterate through valid_divisors in increasing order.
    # This helps keep current_max low, allowing deeper search.
    
    def dfs(rem_prod, min_next, current_max, count, path):
        nonlocal best_max, best_solution
        
        if rem_prod == 1:
            if current_max < best_max:
                best_max = current_max
                best_solution = path[:]
            return
        
        if count >= 12:
            return
            
        # Pruning: 
        # If we must pick one more factor, it must be rem_prod.
        # If rem_prod < min_next, we can't pick it (distinctness/ordering).
        # If rem_prod >= best_max, picking it won't improve (unless it's the last one and < best_max? No, if rem_prod is the last factor, max becomes max(current_max, rem_prod). If rem_prod >= best_max, no improvement).
        
        # General Pruning:
        # The maximum factor in any valid improvement must be < best_max.
        # So, any factor we pick must be < best_max.
        # Also, the remaining product must be factorable into distinct integers < best_max.
        
        # If rem_prod is prime:
        # It must be picked as a single factor (cannot be split into integers > 1).
        # So if count + 1 > 12, fail.
        # If rem_prod < min_next, fail.
        # If rem_prod >= best_max, fail (won't improve).
        # Else, pick it.
        
        # Heuristic: Check if rem_prod can be formed by remaining slots with factors < best_max.
        # Max product of k distinct integers < M is approx (M-1)^k.
        # This is a loose bound.
        
        # Iterate through candidates
        # We only care about divisors of rem_prod.
        # We iterate through valid_divisors.
        # To avoid iterating all 10^5 divisors for every step, we can be smart.
        # But 10^5 * 12 (depth) * branching factor?
        # Branching factor is number of divisors of rem_prod.
        # For 12!, number of divisors is 6615.
        # Depth is up to 12.
        # 6615^12 is huge, but pruning is strong.
        
        # Optimization: 
        # Only consider d if d divides rem_prod.
        # And d >= min_next.
        # And d < best_max (to ensure potential improvement).
        
        for d in valid_divisors:
            if d < min_next:
                continue
            if d >= best_max:
                break # Since sorted, no larger d will improve max
            if rem_prod % d != 0:
                continue
            
            # If we pick d, new rem is rem_prod // d
            new_rem = rem_prod // d
            new_max = max(current_max, d)
            
            # If new_max >= best_max, we can't improve?
            # Wait, if new_max < best_max, we might still improve.
            # If new_max >= best_max, then the final max will be >= new_max >= best_max.
            # So we can prune this branch.
            if new_max >= best_max:
                # Since d is increasing, new_max will only increase or stay same.
                # But current_max is fixed. 
                # If d < current_max, new_max = current_max.
                # If current_max >= best_max, we would have pruned earlier?
                # No, current_max is from previous steps.
                # If current_max >= best_max, we shouldn't be here?
                # Actually, if current_max >= best_max, any extension will have max >= current_max >= best_max.
                # So we can check at the start of dfs: if current_max >= best_max: return.
                
                # Let's add that check at the top of dfs.
                pass
            
            # Additional pruning: 
            # If new_rem == 1, we found a solution.
            # Else, we need to factor new_rem into (12 - count - 1) factors.
            # All factors must be distinct, > d, and < best_max.
            
            # If new_rem is prime:
            if new_rem > 1:
                # Check if new_rem is prime
                # If prime, it must be the last factor.
                # So we need count + 2 <= 12.
                # And new_rem >= d + 1 (distinctness).
                # And new_rem < best_max.
                # If not prime, it can be split.
                
                # Simple prime check for small numbers
                def is_prime(n):
                    if n < 2: return False
                    if n == 2: return True
                    if n % 2 == 0: return False
                    for i in range(3, int(math.isqrt(n)) + 1, 2):
                        if n % i == 0:
                            return False
                    return True

                if is_prime(new_rem):
                    if count + 2 > 12:
                        continue
                    if new_rem <= d:
                        continue
                    if new_rem >= best_max:
                        continue
                    # Pick new_rem as the next factor
                    path.append(new_rem)
                    dfs(1, new_rem + 1, max(new_max, new_rem), count + 2, path)
                    path.pop()
                    continue
            
            # Recurse
            path.append(d)
            dfs(new_rem, d + 1, new_max, count + 1, path)
            path.pop()
            
    # Initial call
    # We can start with count=0, min_next=13, current_max=1
    # But we must pick at least 1 factor.
    dfs(target, 13, 1, 0, [])
    
    return best_solution
