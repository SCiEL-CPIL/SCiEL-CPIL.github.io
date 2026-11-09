import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    # 1. Calculate 12!
    target = math.factorial(12)  # 479001600
    
    # 2. Generate all divisors of target efficiently using prime factorization
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
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
    
    # Precompute which divisors divide which other divisors for faster lookup
    # Actually, let's use a different approach: iterative deepening on the max factor
    
    # We want to find the minimum M such that 12! can be factored into 
    # 1-12 distinct integers all in (12, M]
    
    # Binary search or linear search on M
    # Lower bound: at least 13
    # Upper bound: target (single factor)
    
    # Let's try values of M from low to high, checking if a valid decomposition exists
    
    # For efficiency, let's use DFS with memoization on (remaining, start_idx, num_factors)
    # But the state space might be large. Let's use a smarter approach.
    
    # Key insight: we can do DFS where we pick factors in increasing order,
    # and prune aggressively based on the current maximum allowed.
    
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
        
        # If remaining > max_allowed, we can't use it as a single factor,
        # but we could potentially factor it further. However, since all factors
        # must be <= max_allowed, if remaining > max_allowed, we need to factor it
        # into multiple factors each <= max_allowed.
        
        # Pruning: if remaining is prime and > max_allowed, fail
        # More generally, if remaining cannot be factored into factors in [valid_divs[start_idx], max_allowed]
        
        # Find valid divisors of remaining in range [valid_divs[start_idx], min(max_allowed, remaining)]
        upper_bound = min(max_allowed, remaining)
        
        for i in range(start_idx, n_divs):
            d = valid_divs[i]
            if d > upper_bound:
                break
            if remaining % d != 0:
                continue
            
            # Try using d as the next factor
            factors_so_far.append(d)
            result = can_factor(remaining // d, i + 1, max_allowed, factors_so_far, max_factors)
            if result is not None:
                return result
            factors_so_far.pop()
            
            if time.time() > deadline:
                return None
        
        return None
    
    # Search for the minimum max_allowed
    # We can binary search on max_allowed, but linear from low might be faster if the answer is small
    
    # What's a reasonable lower bound? 
    # 12! = 479001600. If we use k factors each <= M, then M^k >= 12!
    # For k=12: M^12 >= 479001600 => M >= 479001600^(1/12) ≈ 12.9, so M >= 13
    # For k=10: M^10 >= 479001600 => M >= 479001600^(1/10) ≈ 14.8, so M >= 15
    # For k=8: M^8 >= 479001600 => M >= 479001600^(1/8) ≈ 17.5, so M >= 18
    # For k=6: M^6 >= 479001600 => M >= 479001600^(1/6) ≈ 25.4, so M >= 26
    # For k=4: M^4 >= 479001600 => M >= 479001600^(1/4) ≈ 83.1, so M >= 84
    # For k=3: M^3 >= 479001600 => M >= 479001600^(1/3) ≈ 782.9, so M >= 783
    # For k=2: M^2 >= 479001600 => M >= 479001600^(1/2) ≈ 21886, so M >= 21887
    
    # Let's try to find a good solution by trying various max_factors and max_allowed
    
    best_result = [target]
    best_max = target
    
    # Try different numbers of factors, from most to least (more factors allow smaller max)
    for max_factors in range(12, 0, -1):
        # For this number of factors, find the minimum max_allowed
        # Binary search on max_allowed
        lo = 13
        hi = target
        found = None
        
        # First, check if hi works
        if time.time() > deadline:
            break
        
        while lo <= hi and time.time() < deadline:
            mid = (lo + hi) // 2
            
            # Check if we can factor target into max_factors distinct factors in (12, mid]
            factors_so_far = []
            result = can_factor(target, 0, mid, factors_so_far, max_factors)
            
            if result is not None:
                found = result
                hi = mid - 1
            else:
                lo = mid + 1
        
        if found is not None:
            cur_max = max(found)
            if cur_max < best_max:
                best_max = cur_max
                best_result = found
            # Since we're going from more factors to fewer, and we want to minimize max,
            # a solution with more factors might have a smaller max. But we should continue
            # to check fewer factors too, in case they give an even smaller max (unlikely but possible).
    
    return best_result
