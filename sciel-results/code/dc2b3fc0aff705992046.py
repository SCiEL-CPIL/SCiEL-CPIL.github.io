import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    # 1. Calculate 12!
    target = math.factorial(12)  # 479001600
    
    # 2. Generate all divisors of target
    divisors = set()
    for a in range(11):      # 2^a
        for b in range(6):   # 3^b
            for c in range(3): # 5^c
                for d in range(2): # 7^d
                    for e in range(2): # 11^e
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        divisors.add(val)
    
    # Filter: must be > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    n_divs = len(valid_divs)
    
    # Precompute logs for potential balancing heuristics if needed, 
    # but the multiplicative bounds are stronger.
    
    def can_factor(remaining, start_idx, max_allowed, factors_so_far, max_factors):
        """
        Check if we can factor 'remaining' into (max_factors - len(factors_so_far))
        distinct factors, each > 12, >= valid_divs[start_idx], and <= max_allowed.
        """
        if time.time() > deadline:
            return None
        
        if remaining == 1:
            if 1 <= len(factors_so_far) <= 12:
                return factors_so_far[:]
            return None
        
        # We need at least 1 more factor
        if len(factors_so_far) >= max_factors:
            return None
        
        num_needed = max_factors - len(factors_so_far)
        
        # Pruning 1: Minimum possible product for num_needed factors
        # The smallest distinct integers > 12 are 13, 14, ...
        # However, they must also be >= valid_divs[start_idx].
        # Let's just use the theoretical min: 13 * 14 * ... * (12+num_needed)
        # But we must respect the 'start_idx' constraint which enforces increasing order.
        # A simpler lower bound: 13^num_needed is too low. 
        # Let's use the actual divisors if possible, but that's expensive.
        # Let's use: min_factor = valid_divs[start_idx] if start_idx < n_divs else 13
        # Actually, the factors must be distinct and increasing.
        # The smallest valid next factor is valid_divs[start_idx].
        # The next smallest is valid_divs[start_idx+1], etc.
        # If start_idx + num_needed > n_divs, we can't pick enough distinct divisors.
        if start_idx + num_needed > n_divs:
            return None
            
        # Calculate the minimum possible product using the smallest available distinct divisors
        # Note: These divisors must divide 'remaining'. This check is optimistic (ignores divisibility).
        min_prod = 1
        for j in range(num_needed):
            min_prod *= valid_divs[start_idx + j]
            if min_prod > remaining:
                return None
        
        # Pruning 2: Maximum possible product for num_needed factors
        # The largest possible factors are max_allowed, max_allowed-1, ... 
        # But they must be divisors of 'remaining' and distinct.
        # Optimistic bound: max_allowed * (max_allowed - 1) * ... * (max_allowed - num_needed + 1)
        max_prod = 1
        for j in range(num_needed):
            max_prod *= (max_allowed - j)
            if max_prod < remaining:
                return None
                
        # Pruning 3: If remaining is prime and > max_allowed, fail
        # (Check if remaining has any divisor in range)
        
        # Find valid divisors of remaining in range [valid_divs[start_idx], min(max_allowed, remaining)]
        upper_bound = min(max_allowed, remaining)
        
        for i in range(start_idx, n_divs):
            d = valid_divs[i]
            if d > upper_bound:
                break
            if remaining % d != 0:
                continue
            
            # Additional prune: After picking d, can we form remaining//d with num_needed-1 factors?
            # The next factors must be > d (distinct, increasing) and <= max_allowed.
            # Min product for next: (d+1) * (d+2) * ... (d + num_needed - 1) ? No, they must be divisors.
            # Let's just recurse.
            
            factors_so_far.append(d)
            result = can_factor(remaining // d, i + 1, max_allowed, factors_so_far, max_factors)
            if result is not None:
                return result
            factors_so_far.pop()
            
            if time.time() > deadline:
                return None
        
        return None

    # Search for the minimum max_allowed
    # We know M must be at least 13.
    # Let's find a tight lower bound.
    # If we use k factors, M^k >= 12! => M >= (12!)^(1/k).
    # We want to minimize M. The minimum M will likely be achieved with a specific k.
    # Let's iterate M from a reasonable lower bound upwards.
    
    # What is the absolute lower bound for M?
    # If k=12, M >= 13.
    # If k=11, M >= 14.
    # ...
    # Let's just start from 13 and go up.
    # The number of divisors is small (~600). The search space is constrained.
    
    best_result = [target]
    best_max = target
    
    # We can binary search M, but linear might be safer if the optimal M is small.
    # However, binary search is O(log N) checks. Linear is O(N).
    # Given the strong pruning, each check is fast.
    # Let's use binary search on M in [13, target].
    
    lo = 13
    hi = target
    found_solution = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        factors_so_far = []
        result = can_factor(target, 0, mid, factors_so_far, 12)
        
        if result is not None:
            found_solution = result
            best_max = max(found_solution)
            best_result = found_solution
            hi = mid - 1
        else:
            lo = mid + 1
            
    return best_result
