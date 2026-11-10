"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    deadline = time.time() + 82.0
    N = math.factorial(12)
    
    # Prime exponents for 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target_exp = (10, 5, 2, 1, 1)
    primes = [2, 3, 5, 7, 11]
    
    def get_exponents(n):
        exps = [0]*5
        for i, p in enumerate(primes):
            while n % p == 0:
                exps[i] += 1
                n //= p
        return tuple(exps)
    
    # Precompute ALL divisors of 12! that are > 12
    all_divisors = []
    for d in range(13, N + 1):
        if N % d == 0:
            all_divisors.append(d)
    all_divisors.sort()
    all_div_exps = [get_exponents(d) for d in all_divisors]
    
    def can_partition(max_factor):
        # Get divisors <= max_factor
        # Binary search to find the cutoff
        import bisect
        cutoff = bisect.bisect_right(all_divisors, max_factor)
        divs = all_divisors[:cutoff]
        div_exps = all_div_exps[:cutoff]
        n_divs = len(divs)
        
        if n_divs == 0:
            return None
        
        # DFS with aggressive pruning
        # We'll use a recursive approach with memoization on (idx, rem_exp)
        # Key insight: order matters - we should try to use smaller divisors first
        # but also prune based on whether remaining product is achievable
        
        # For pruning: compute the max product achievable with remaining divisors
        # and min product needed
        # Actually, let's use a different approach: 
        # At each step, we need the product of chosen divisors to equal N
        # Track remaining product as an integer
        
        # Better approach: track remaining exponent vector
        # Pruning: 
        # 1. If remaining product > product of all remaining divisors -> impossible
        # 2. If remaining product < 1 -> impossible (shouldn't happen)
        # 3. If we've used 12 factors -> impossible unless done
        
        # Let's use a simpler but effective approach:
        # Try subsets in a smart order, with strong pruning
        
        # Precompute: for each position i, the max product of any subset of divs[i:] 
        # that we could still use (limited to 12-count factors)
        # This is complex, so let's use a simpler bound:
        # The product of the largest (12-count) divisors from position i onwards
        
        # Actually, let's just do DFS with good ordering and basic pruning
        # The key improvement: try taking divisors first (greedy bias toward using factors)
        # and prune when remaining product can't be achieved
        
        @lru_cache(maxsize=None)
        def dfs(idx, rem_exp, count):
            if time.time() > deadline:
                return None
            
            if rem_exp == (0, 0, 0, 0, 0):
                if count <= 12:
                    return []
                return None
            
            if count >= 12:
                return None
            
            if idx >= n_divs:
                return None
            
            # Pruning: check if remaining exponents are non-negative
            for i in range(5):
                if rem_exp[i] < 0:
                    return None
            
            # Pruning: compute the minimum and maximum product achievable
            # with the remaining divisors (divs[idx:]) using at most (12-count) more factors
            # Min product: product of smallest (12-count) divisors from idx onwards
            # Max product: product of largest (12-count) divisors from idx onwards
            
            remaining_slots = 12 - count
            if remaining_slots <= 0:
                return None
            
            # Simple pruning: the product of rem_exp must be achievable
            # rem_product = 2^rem_exp[0] * 3^rem_exp[1] * 5^rem_exp[2] * 7^rem_exp[3] * 11^rem_exp[4]
            rem_prod = 1
            for i in range(5):
                rem_prod *= primes[i] ** rem_exp[i]
            
            # Max product achievable with remaining_slots factors from divs[idx:]
            # Take the largest remaining_slots divisors
            if n_divs - idx >= remaining_slots:
                max_avail = 1
                for j in range(n_divs - 1, n_divs - 1 - remaining_slots, -1):
                    max_avail *= divs[j]
            else:
                max_avail = 1
                for j in range(idx, n_divs):
                    max_avail *= divs[j]
            
            if rem_prod > max_avail:
                return None
            
            # Min product: if we must use at least 1 more factor (since rem_exp != 0)
            # The smallest divisor from idx onwards
            if rem_prod < divs[idx]:
                # We need at least one more factor, and the smallest available is divs[idx]
                # But we could skip it... actually we need the product to be exactly rem_prod
                # If rem_prod < divs[idx], we can't use any divisor from idx onwards
                # because all are >= divs[idx] > rem_prod
                # But wait, we could use a divisor that's a factor of rem_prod
                # Actually no - we need the PRODUCT of chosen divisors to be rem_prod
                # Each divisor must divide rem_prod
                # So if no divisor from idx onwards divides rem_prod, we're stuck
                # But let's just let the recursion handle it
                pass
            
            # Option 1: take current divisor if it fits
            d_exp = div_exps[idx]
            new_rem = tuple(rem_exp[i] - d_exp[i] for i in range(5))
            
            valid = True
            for i in range(5):
                if new_rem[i] < 0:
                    valid = False
                    break
            
            if valid:
                sub_result = dfs(idx + 1, new_rem, count + 1)
                if sub_result is not None:
                    return [divs[idx]] + sub_result
            
            # Option 2: skip current divisor
            result = dfs(idx + 1, rem_exp, count)
            if result is not None:
                return result
            
            return None
        
        result = dfs(0, target_exp, 0)
        return result
    
    # Binary search for minimum max_factor
    # Lower bound: 13
    # Upper bound: let's use a tighter bound
    # We know a valid solution exists with max factor around a few hundred
    # Let's start with hi = 1000 and expand if needed
    lo = 13
    hi = 200
    best_solution = None
    
    # First check if hi is sufficient
    sol = can_partition(hi)
    if sol is None:
        # Expand hi
        hi = 1000
        sol = can_partition(hi)
        if sol is None:
            hi = 5000
            sol = can_partition(hi)
            if sol is None:
                hi = N
    
    if sol is not None:
        best_solution = sol
    
    # Now binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol = can_partition(mid)
        
        if sol is not None:
            best_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_solution is None:
        # Fallback: return a known valid solution
        return [20, 27, 28, 30, 32, 33]
    
    return best_solution
# EVOLVE-BLOCK-END
