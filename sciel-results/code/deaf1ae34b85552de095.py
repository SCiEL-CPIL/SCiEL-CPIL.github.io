import time
from math import factorial

_START = time.time()
_DEADLINE = _START + 85

def construct():
    global _START, _DEADLINE
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = []
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.append(val)
    
    divisors.sort()
    divisor_set = set(divisors)
    
    # Precompute: for each divisor, list of divisors that divide it (and are > 12)
    # This will help with faster lookup
    
    # Strategy: DFS with strong pruning
    # Key insight: if remaining product is R, and we've used factors up to min_val,
    # then the next factor must be >= min_val, and R/factor must be expressible
    # as product of distinct factors >= factor (or equal to 1)
    
    # Upper bound on max factor: start with a reasonable guess
    # (12!)^(1/6) ≈ 55, let's try to do better
    # Let's binary search or just do DFS with pruning
    
    best_max = target
    best_factors = [target]
    
    # Precompute divisors_of: for each value that could be 'remaining',
    # what divisors of it are in our valid set?
    # But remaining can be many values... let's compute on the fly but cache.
    
    from functools import lru_cache
    
    # For speed, let's precompute which divisors divide which other divisors
    # divisors[i] divides divisors[j] check
    
    # Actually, let's use a different strategy:
    # The number of divisors is 11*6*3*2*2 = 792, but filtered > 12 it's less.
    # Let's count: 792 total divisors, minus those <= 12.
    # Divisors <= 12: 1,2,3,4,5,6,7,8,9,10,11,12 = 12 values
    # So 792 - 12 = 780 valid divisors. That's a lot to iterate.
    
    # Better approach: at each DFS level, only consider divisors of 'remaining'
    # that are >= min_allowed. We can compute these quickly.
    
    def get_valid_divisors_of(n, min_val):
        """Get divisors of n that are >= min_val and in divisor_set"""
        result = []
        # n divides target, so its divisors are a subset of target's divisors
        # Iterate through divisors >= min_val and check if they divide n
        for d in divisors:
            if d < min_val:
                continue
            if d > n:
                break
            if n % d == 0:
                result.append(d)
        return result
    
    # To make this faster, let's precompute a mapping
    # Actually with 780 divisors, iterating through them at each level is O(780) per node
    # The tree depth is at most 12, so worst case is 780^12 which is way too much.
    # But pruning will cut it down enormously.
    
    # Let's add stronger pruning:
    # 1. If remaining == 1: success
    # 2. If remaining is prime (in the sense that it has no valid divisors >= min_val 
    #    other than itself, and itself >= min_val): then we must use remaining as the last factor
    # 3. If remaining < min_val: fail (can't form product)
    # 4. The next factor d must satisfy: d >= min_val, d divides remaining,
    #    and remaining/d must be >= 1 and if > 1, must be expressible with factors >= d
    
    # Additional pruning: the product of the smallest k factors >= min_val that divide
    # remaining should be <= remaining. But this is complex.
    
    # Simpler: if remaining is a valid divisor and remaining >= min_val, 
    # we can always use it as the last factor.
    
    # Let's also add: if we have used 'used_count' factors, we can use at most 
    # 12 - used_count more factors. The minimum product of (12-used_count) distinct 
    # integers >= min_val is min_val * (min_val+1) * ... but they must divide remaining.
    
    # For now, let's do a clean DFS with time checking:
    
    def dfs(remaining, min_allowed, current_factors, max_so_far):
        nonlocal best_max, best_factors
        
        if time.time() > _DEADLINE:
            return
        
        if remaining == 1:
            if len(current_factors) >= 1 and max_so_far < best_max:
                best_max = max_so_far
                best_factors = list(current_factors)
            return
        
        # remaining > 1, need at least one more factor
        # The next factor d must satisfy:
        #   d >= min_allowed
        #   d divides remaining
        #   d is in divisor_set (guaranteed if d divides target and d > 12)
        #   remaining // d >= 1
        #   If remaining // d > 1, it must be formable by distinct factors >= d
        #   If remaining // d == 1, we're done with this factor
        
        # Quick check: if remaining itself is >= min_allowed and in divisor_set,
        # we can use it as the last factor
        if remaining >= min_allowed and remaining in divisor_set:
            new_max = max(max_so_far, remaining)
            if new_max < best_max:
                current_factors.append(remaining)
                dfs(1, remaining + 1, current_factors, new_max)
                current_factors.pop()
        
        # Try all valid divisors d of remaining, d >= min_allowed, d < remaining
        # (if d == remaining, already handled above)
        # To ensure we find solutions with small max, try smaller d first
        
        # Optimization: only try d where remaining//d can potentially be formed
        # remaining//d must be >= 1, and if > 1, must have a factorization with factors >= d
        
        # Upper bound on d: if we want max_so_far to stay < best_max, 
        # then d < best_max
        max_d = min(remaining, best_max - 1) if best_max < remaining else remaining
        
        for d in divisors:
            if time.time() > _DEADLINE:
                return
            
            if d < min_allowed:
                continue
            if d > max_d:
                break
            if d >= remaining:
                break  # d == remaining handled above, d > remaining impossible
            if remaining % d != 0:
                continue
            
            new_remaining = remaining // d
            new_max = max(max_so_far, d)
            
            # Pruning: new_remaining must be >= 1
            if new_remaining < 1:
                continue
            
            # Pruning: if new_remaining > 1, it must be expressible as product of
            # distinct factors >= d+1 (since we need distinct factors and we're picking in order)
            # Actually, factors must be >= d (but distinct, so next is >= d+1? No, 
            # distinct means all different, and we're picking in increasing order, so next >= d+1)
            # Wait, we're picking in increasing order to ensure distinctness. So next factor >= d+1.
            # But new_remaining could be 1, in which case we're done.
            
            if new_remaining > 1:
                # new_remaining must be formable by distinct factors >= d+1
                # Quick check: the smallest possible factor is d+1, so new_remaining must be >= d+1
                # Actually, new_remaining could be a single factor >= d+1
                # Or multiple factors, each >= d+1, distinct
                # Minimum product of 1 factor: d+1
                # So if new_remaining < d+1, fail
                if new_remaining < d + 1:
                    continue
                
                # Also, new_remaining must be divisible by some valid factors >= d+1
                # This is checked in the recursive call
            
            current_factors.append(d)
            dfs(new_remaining, d + 1, current_factors, new_max)
            current_factors.pop()
    
    dfs(target, 13, [], 0)
    
    if best_factors == [target]:
        # Fallback: try a heuristic approach
        # Greedily pick small factors
        pass
    
    return best_factors
