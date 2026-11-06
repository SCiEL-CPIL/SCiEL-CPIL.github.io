"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 12
    target = math.factorial(N)  # 479001600
    min_factor = N + 1  # 13
    
    # Precompute all divisors of target, sorted
    all_divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            all_divisors.append(i)
            if i != target // i:
                all_divisors.append(target // i)
    all_divisors.sort()
    
    # Filter: only divisors > 12
    valid_divisors = [d for d in all_divisors if d > 12]
    
    # Precompute a set for O(1) membership
    valid_div_set = set(valid_divisors)
    
    # Suffix product array: suffix_prod[i] = product of valid_divisors[i:]
    # This gives us a lower bound: if remaining < suffix_prod[i], we can't
    # form remaining using distinct factors from valid_divisors[i:] onward.
    # But we need factors >= min_allowed, so we find the index of the first
    # valid divisor >= min_allowed.
    # Actually, the constraint is that factors must be strictly increasing
    # (since they're distinct and we pick in order). So if we've picked factor d,
    # next must be > d.
    # 
    # Better approach: precompute suffix products of valid_divisors.
    # suffix_prod[i] = valid_divisors[i] * valid_divisors[i+1] * ... * valid_divisors[-1]
    # But this product can be astronomically large. We only need it up to a point.
    # Let's compute suffix_prod with capping: if product > target, cap at target+1.
    
    n_divs = len(valid_divisors)
    suffix_prod = [1] * (n_divs + 1)
    for i in range(n_divs - 1, -1, -1):
        suffix_prod[i] = suffix_prod[i + 1] * valid_divisors[i]
        if suffix_prod[i] > target:
            suffix_prod[i] = target + 1  # cap to avoid huge numbers
    
    # Also precompute prefix: for a given min_allowed, find index in valid_divisors
    # where valid_divisors[idx] >= min_allowed
    # We'll use bisect for this.
    import bisect
    
    # Get an initial upper bound via a greedy approach
    # Try to factor target into as many small distinct factors > 12 as possible
    def greedy_upper_bound():
        """Try to find a decent solution greedily."""
        remaining = target
        factors = []
        min_f = 13
        while remaining > 1 and len(factors) < 12:
            # Find the smallest divisor of remaining that is >= min_f
            found = None
            for d in valid_divisors:
                if d < min_f:
                    continue
                if d >= remaining:
                    break
                if remaining % d == 0:
                    found = d
                    break
            if found is None:
                # No divisor found; try the remaining itself if it's > min_f
                if remaining > min_f and remaining in valid_div_set:
                    found = remaining
                else:
                    return None
            factors.append(found)
            remaining //= found
            min_f = found + 1
        if remaining == 1:
            return max(factors)
        return None
    
    ub = greedy_upper_bound()
    if ub is None:
        ub = target
    else:
        ub = min(ub, target)
    
    best_largest = [ub]
    best_factors = [target]  # placeholder
    
    # DFS with strong pruning
    def dfs(remaining, min_allowed, factors_so_far, current_max, depth):
        if time.time() > deadline:
            return
        
        # Prune: if current_max already >= best, no improvement possible
        if current_max >= best_largest[0]:
            return
        
        # Base case
        if remaining == 1:
            if factors_so_far:
                if current_max < best_largest[0]:
                    best_largest[0] = current_max
                    best_factors[:] = factors_so_far[:]
            return
        
        # If we've used 12 factors already and remaining != 1, dead end
        if len(factors_so_far) >= 12:
            return
        
        # Quick prune: remaining must be >= min_allowed
        if remaining < min_allowed:
            return
        
        # Find the index in valid_divisors where we start looking
        idx = bisect.bisect_left(valid_divisors, min_allowed)
        
        # Suffix product prune: the product of the smallest available factors
        # (starting from idx) must be <= remaining, otherwise we can't form
        # remaining with distinct factors from this point.
        # Actually, we need: there exists a subset of {valid_divisors[idx], ...}
        # whose product equals remaining. The minimum product of any non-empty
        # subset starting at idx is valid_divisors[idx].
        # A stronger prune: the product of ALL remaining valid divisors from idx
        # must be >= remaining (otherwise even using all of them, we can't reach remaining).
        if suffix_prod[idx] < remaining:
            return
        
        # Also: remaining must be achievable. The minimum factor we can pick is
        # valid_divisors[idx]. If that doesn't divide remaining, we still might
        # pick a larger one. So we iterate.
        
        # Iterate through valid divisors starting from idx
        for i in range(idx, n_divs):
            d = valid_divisors[i]
            
            # Prune: d >= best means all subsequent are too
            if d >= best_largest[0]:
                break
            
            # Prune: d must divide remaining
            if remaining % d != 0:
                continue
            
            quotient = remaining // d
            
            # If quotient == 1, we're done with this factor
            if quotient == 1:
                new_max = max(current_max, d)
                if new_max < best_largest[0]:
                    best_largest[0] = new_max
                    best_factors[:] = factors_so_far + [d]
                continue
            
            # The next factor must be > d, so >= d+1
            next_min = d + 1
            
            # Prune: quotient must be >= next_min
            if quotient < next_min:
                continue
            
            # Suffix product prune for the sub-problem:
            # The product of all valid divisors >= next_min must be >= quotient
            next_idx = bisect.bisect_left(valid_divisors, next_min)
            if suffix_prod[next_idx] < quotient:
                continue
            
            # Also: if we've already used many factors, check we don't exceed 12
            if len(factors_so_far) + 1 >= 12:
                # We can only add one more factor, so quotient must be 1
                # (already handled above) or we need exactly one more factor = quotient
                # which must be > d and distinct
                if quotient in valid_div_set and quotient > d and quotient < best_largest[0]:
                    new_max = max(current_max, d, quotient)
                    if new_max < best_largest[0]:
                        best_largest[0] = new_max
                        best_factors[:] = factors_so_far + [d, quotient]
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_largest[0]:
                continue
            
            factors_so_far.append(d)
            dfs(quotient, next_min, factors_so_far, new_max, depth + 1)
            factors_so_far.pop()
    
    # Start DFS
    dfs(target, min_factor, [], 0, 0)
    
    return best_factors[:]
# EVOLVE-BLOCK-END
