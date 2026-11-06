"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    target = factorial(12)
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def can_decompose(remaining, max_factor, used_set, min_factor=13):
        """
        Try to decompose 'remaining' into distinct factors, each in [min_factor, max_factor].
        Returns a list of factors or None.
        Uses backtracking, trying larger factors first to minimize the max.
        """
        if remaining == 1:
            return []
        
        # If remaining > max_factor, we can't fit it in one factor, need more factors
        # But if remaining itself is > max_factor, we need to split it
        
        # Try factors from max_factor down to min_factor
        # We need f to divide remaining, f not in used_set
        # To be efficient, iterate divisors
        
        # Get divisors of remaining that are in [min_factor, max_factor]
        best_result = None
        
        # We need to find a factor f of remaining where min_factor <= f <= max_factor
        # and f not in used_set
        # Try from max_factor downward to encourage smaller max
        
        # But we also need the remaining/remaining after picking f to be decomposable
        # with factors >= min(remaining/f, max_factor) but actually the remaining factors
        # also need to be in [min_factor, max_factor]
        
        # For efficiency, let's iterate potential factors
        # We need f | remaining, min_factor <= f <= max_factor, f not in used_set
        
        # Find all valid divisors
        # remaining can be large, so let's be smart
        
        # Actually, let's iterate f from max_factor down to min_factor
        # and check if f divides remaining
        # But that could be slow if max_factor is large
        
        # Better: find divisors of remaining in range
        # For a number n, divisors come in pairs (d, n/d)
        
        divisors = []
        i = 1
        while i * i <= remaining:
            if remaining % i == 0:
                d1 = i
                d2 = remaining // i
                if min_factor <= d1 <= max_factor and d1 not in used_set:
                    divisors.append(d1)
                if d2 != d1 and min_factor <= d2 <= max_factor and d2 not in used_set:
                    divisors.append(d2)
            i += 1
        
        # Sort descending to try larger factors first (this helps minimize the max in the end)
        divisors.sort(reverse=True)
        
        for f in divisors:
            new_remaining = remaining // f
            # The remaining product new_remaining must be decomposable into factors
            # each in [min_factor, max_factor], distinct from used_set ∪ {f}
            # But also, if new_remaining > 1, it must be >= min_factor (since it'll be at least one factor)
            # Actually new_remaining could be 1 (done) or >= min_factor
            
            if new_remaining == 1:
                return [f]
            
            # new_remaining must be decomposable
            # All factors of new_remaining must be >= min_factor and <= max_factor
            # So new_remaining itself, if it's going to be a single factor, must be <= max_factor
            # If it needs multiple factors, each must be <= max_factor
            
            # Quick check: if new_remaining > max_factor, it needs to be split further
            # If new_remaining <= max_factor and new_remaining >= min_factor and new_remaining not in used_set and new_remaining != f:
            #   then we could use it as a single factor
            
            # Let's just recurse
            new_used = used_set | {f}
            result = can_decompose(new_remaining, max_factor, new_used, min_factor)
            if result is not None:
                return [f] + result
        
        return None
    
    # Binary search on max_factor M
    # Lower bound: at least 13 (must be > 12)
    # Upper bound: target itself (single factor)
    
    # But we want to minimize M, so binary search
    lo = 13
    hi = target
    best_result = [target]  # fallback
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        result = can_decompose(target, mid, set(), 13)
        if result is not None:
            best_result = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_result
# EVOLVE-BLOCK-END
