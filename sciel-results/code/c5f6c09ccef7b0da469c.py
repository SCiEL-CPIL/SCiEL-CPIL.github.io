import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def get_small_part(x):
        """Extract the factor of x composed of primes in 'primes'."""
        factor = 1
        for p in primes:
            while x % p == 0:
                factor *= p
                x //= p
        return factor
    
    def count_distinct_factors(n):
        """Count distinct factors extracted from n+1 to n+24."""
        distinct_factors = set()
        for i in range(1, 25):
            val = n + i
            sp = get_small_part(val)
            if sp > 1:
                distinct_factors.add(sp)
        return len(distinct_factors)
    
    best_n = 1000
    best_score = float('inf')
    
    # 1. Random Sampling Phase
    # Sample random n in [1000, 1000000]
    # If a good candidate is found, refine its neighborhood.
    
    # Initial check on 1000
    score_1000 = count_distinct_factors(1000)
    if score_1000 < best_score:
        best_score = score_1000
        best_n = 1000
    
    num_samples = 0
    while time.time() < deadline - 5.0:  # Leave 5s for refinement
        n_candidate = random.randint(1000, 1000000)
        score = count_distinct_factors(n_candidate)
        num_samples += 1
        
        if score < best_score:
            best_score = score
            best_n = n_candidate
            
            # If we find a very good score (e.g., <= 3), do intensive local search
            if best_score <= 3:
                # Local refinement around best_n
                # Check a window around best_n
                window = 100
                for i in range(-window, window + 1):
                    if time.time() > deadline - 2.0:
                        break
                    nn = best_n + i
                    if 1000 <= nn <= 1000000:
                        s = count_distinct_factors(nn)
                        if s < best_score:
                            best_score = s
                            best_n = nn
                # If we hit a very low score, we can stop early or continue briefly
                if best_score <= 2:
                    # Try to find 1 or 0 if possible, but 2 is already very good
                    # Let's do a few more random samples to see if we can beat it
                    pass
        
        # If we have found a score of 1 or 0, it's likely optimal or near-optimal
        # The minimum possible is 0 (if all n+i are composed only of primes >= 24? No, n+i must have factors. 
        # Wait, "extract its complete factor composed of primes STRICTLY BELOW 24". 
        # If n+i is prime >= 24, its small part is 1. 
        # If all 24 numbers have small part 1, distinct count is 0? 
        # But n+i are consecutive integers. One is divisible by 2, one by 3, etc.
        # So small part will never be 1 for all.
        # Min distinct factors is likely small, e.g., 2 or 3.
        
    # 2. Final Local Refinement on the global best found
    if time.time() < deadline - 1.0:
        # Refine around the best_n found in random sampling
        # Expand window if time permits
        for window in [10, 50, 200, 1000]:
            if time.time() > deadline - 1.0:
                break
            for i in range(-window, window + 1):
                if time.time() > deadline - 1.0:
                    break
                nn = best_n + i
                if 1000 <= nn <= 1000000:
                    s = count_distinct_factors(nn)
                    if s < best_score:
                        best_score = s
                        best_n = nn

    return best_n
# EVOLVE-BLOCK-END
