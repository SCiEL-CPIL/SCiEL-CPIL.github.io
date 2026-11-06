import math
import time
from functools import lru_cache

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    LIMIT = 10**18
    
    # First 16 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    def get_divisors_sorted(factors):
        """Generate all divisors sorted, using iterative construction."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        divs.sort()
        return divs
    
    def tau_perp_from_factors(factors):
        """Count adjacent coprime pairs in sorted divisors."""
        divs = get_divisors_sorted(factors)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    best_n = 2
    best_score = 0
    
    def try_candidate(factors):
        nonlocal best_n, best_score
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT or n < 2:
            return
        score = tau_perp_from_factors(factors)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Strategy: 
    # Key insight: For maximizing coprime adjacent pairs, we want:
    # 1. Many distinct primes (max k) to create many "support changes"
    # 2. Exponents that create "gaps" in the divisor sequence where supports are disjoint
    # 
    # The product of first 15 primes is ~6.15e17, leaving budget for extra exponents.
    # Let's focus on k=15 and optimize exponents of small primes.
    
    # Precompute: product of first k primes
    prime_products = [1]
    for p in primes:
        prime_products.append(prime_products[-1] * p)
    
    # Max k such that prime_products[k] <= LIMIT
    max_k = 0
    for k in range(len(prime_products)):
        if prime_products[k] <= LIMIT:
            max_k = k
        else:
            break
    
    # Search over k from max_k down to 10
    for k in range(max_k, 9, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        base_product = prime_products[k]
        if base_product > LIMIT:
            continue
            
        remaining_budget = LIMIT // base_product
        
        # We'll assign extra exponents to the first few primes.
        # The key observation: adding exponent to prime p_i creates more divisors
        # that are multiples of p_i, which can create more boundaries.
        # 
        # Let's try a greedy approach: for each prime in order, try increasing
        # its exponent and see if it improves the score.
        
        # Start with all exponents = 1
        current_exponents = [1] * k
        current_n = base_product
        current_score = tau_perp_from_factors([(primes[i], current_exponents[i]) for i in range(k)])
        
        if current_score > best_score:
            best_score = current_score
            best_n = current_n
        
        # Now try to increase exponents of small primes while staying within budget
        # Use a hill-climbing approach: try increasing each exponent by 1, keep if improves
        
        improved = True
        while improved and (time.time() - start_time < DEADLINE):
            improved = False
            for i in range(min(6, k)):  # Only try first 6 primes
                if current_n * primes[i] > LIMIT:
                    continue
                # Try increasing exponent of prime i
                test_exponents = current_exponents.copy()
                test_exponents[i] += 1
                test_n = current_n * primes[i]
                if test_n > LIMIT:
                    continue
                test_score = tau_perp_from_factors([(primes[j], test_exponents[j]) for j in range(k)])
                if test_score > current_score:
                    current_exponents = test_exponents
                    current_n = test_n
                    current_score = test_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
                    break  # Restart from beginning after improvement
        
        # Also try a more systematic search for small k values
        # For k <= 12, we can afford a more exhaustive search
        if k <= 12:
            def search_exponents(idx, current_mult, current_factors_list):
                if time.time() - start_time > DEADLINE:
                    return
                
                if idx == 4:  # We only flex first 4 primes
                    final_factors = []
                    for i in range(k):
                        if i < 4:
                            final_factors.append((primes[i], 1 + current_factors_list[i]))
                        else:
                            final_factors.append((primes[i], 1))
                    try_candidate(final_factors)
                    return
                
                p = primes[idx]
                max_extra = 0
                temp = remaining_budget // current_mult
                while temp >= p:
                    temp //= p
                    max_extra += 1
                
                # Limit search depth for speed
                max_extra = min(max_extra, 15)
                
                for extra in range(max_extra + 1):
                    new_mult = current_mult * (p ** extra)
                    if new_mult > remaining_budget:
                        break
                    search_exponents(idx + 1, new_mult, current_factors_list + [extra])
            
            search_exponents(0, 1, [])
    
    # Fallback
    if best_n == 2 and best_score == 0:
        try_candidate([(p, 1) for p in primes[:15]])
    
    return best_n

# EVOLVE-BLOCK-END
