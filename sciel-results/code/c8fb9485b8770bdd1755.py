"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from math import factorial, log
import time

def _get_divisors(n):
    """Return all divisors of n."""
    divisors = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return divisors

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin from 90s limit
    
    target = factorial(12)
    
    # Get all divisors of target
    all_divs = _get_divisors(target)
    # Only divisors > 12
    valid_divs = sorted([d for d in all_divs if d > 12])
    
    # Precompute suffix products for pruning
    # For a given candidate list, precompute the max product achievable
    # from position i to end (using largest factors)
    
    def find_solution(max_bound, max_factors=12):
        """Find a subset of valid_divs (all <= max_bound) that multiplies to target,
        with at most max_factors elements. Return the list or None."""
        # Get candidates <= max_bound
        candidates = [d for d in valid_divs if d <= max_bound]
        n = len(candidates)
        
        if n == 0:
            return None
        
        # Sort in decreasing order
        candidates.sort(reverse=True)
        
        # Precompute suffix max products: suffix_prod[i] = product of candidates[i:]
        # But this can overflow, so use logs
        suffix_log_prod = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_log_prod[i] = suffix_log_prod[i + 1] + log(candidates[i])
        
        log_target = log(target)
        
        result = []
        
        def backtrack(start, remaining, count):
            if remaining == 1:
                return True
            if count >= max_factors or start >= n:
                return False
            
            # Pruning: even if we take all remaining candidates, can we reach 'remaining'?
            # The max product from start to end is exp(suffix_log_prod[start])
            # We need remaining <= that max product
            if remaining > 1:
                # Check if the product of all remaining candidates (from start) is >= remaining
                # Using logs to avoid overflow
                if suffix_log_prod[start] < log(remaining) - 1e-12:
                    return False
            
            # Also, we need at least (count + 1) factors if remaining > 1
            # and we can use at most (max_factors - count) more factors
            # The minimum factor we can pick is candidates[-1] (smallest)
            # So remaining must be >= candidates[-1]^(max_factors - count) if we need that many
            # Actually, simpler: remaining must be achievable with at most (max_factors - count) factors
            # Each factor >= 13 (since > 12)
            # So remaining >= 13^(max_factors - count + 1) is NOT required, but
            # remaining must be >= 13 if we need at least one more factor
            if remaining < 13:
                # remaining must be 1, but we already checked that
                return False
            
            # Try each candidate from start to n-1
            for i in range(start, n):
                c = candidates[i]
                if remaining % c != 0:
                    continue
                new_remaining = remaining // c
                result.append(c)
                if backtrack(i + 1, new_remaining, count + 1):
                    return True
                result.pop()
                
                # Pruning: if c > remaining, we can break (but c <= max_bound and remaining decreases)
                # Actually, since candidates are sorted descending, if c > remaining, 
                # then all subsequent candidates are <= c but we still need to check divisibility.
                # However, if new_remaining < 13 and count + 1 < max_factors, we need more factors
                # but the smallest factor is 13, so new_remaining must be >= 13 * (something)
                # This is handled by the base case.
            
            return False
        
        if backtrack(0, target, 0):
            return result
        return None
    
    # Iterate M over valid_divs in increasing order
    best_solution = [target]  # Fallback: single factor = 12!
    best_max = target
    
    for M in valid_divs:
        if M >= best_max:
            break
        if time.time() - start_time > timeout:
            break
        
        sol = find_solution(M)
        if sol is not None:
            best_solution = sol
            best_max = M
            break
    
    return best_solution

# EVOLVE-BLOCK-END
