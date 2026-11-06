"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import reduce

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    LIMIT = 10**18
    
    # First 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return sorted(divs)
    
    def tau_perp_from_factors(factors):
        """Count adjacent coprime pairs in sorted divisors."""
        divs = get_divisors_from_factors(factors)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count

    # 1. Generate the squarefree product of the first 15 primes
    primes_15 = primes[:15]
    n_sqfree = 1
    for p in primes_15:
        n_sqfree *= p
    
    candidates = []
    if n_sqfree <= LIMIT:
        # factors: list of (prime, exponent)
        factors_sqfree = [(p, 1) for p in primes_15]
        candidates.append((n_sqfree, factors_sqfree, "squarefree_15"))

    # 2. Find numbers with high number of divisors
    # We use a recursive search to find top candidates by divisor count
    # We store (div_count, n, factors)
    top_candidates = []
    
    def search_max_divisors(idx, max_exp, curr_n, curr_divs, curr_factors):
        if time.time() - start_time > DEADLINE - 5:
            return
        
        # Add current state to candidates if it has enough divisors
        # We want a diverse set of high-divisor numbers
        if curr_divs > 1000: # Threshold to avoid too many small numbers
             top_candidates.append((curr_divs, curr_n, list(curr_factors)))
        
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Try exponents from max_exp down to 0
        # Optimization: if curr_n * p > LIMIT, we can only use exponent 0
        # Actually, we need to check p^e
        e = max_exp
        while e >= 0:
            # Check if curr_n * p^e <= LIMIT
            # To avoid overflow or slow power, we multiply iteratively
            # But since we are in a loop, we can just check
            
            # Calculate new_n
            new_n = curr_n
            # Multiply by p^e
            # We can do this in the loop or precompute
            # Since e is small, this is fast
            temp_n = curr_n
            valid = True
            for _ in range(e):
                if temp_n > LIMIT // p:
                    valid = False
                    break
                temp_n *= p
            
            if not valid:
                e -= 1
                continue
                
            new_divs = curr_divs * (e + 1)
            new_factors = curr_factors + [(p, e)]
            
            # Recurse
            search_max_divisors(idx + 1, e, temp_n, new_divs, new_factors)
            
            e -= 1
            if e < 0:
                break

    # Start search
    # Max exponent for 2 is around 60 (2^60 > 10^18)
    search_max_divisors(0, 60, 1, 1, [])
    
    # Sort top_candidates by divisor count descending
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 20 unique numbers
    seen_ns = set()
    search_candidates = []
    for div_count, n, factors in top_candidates:
        if n in seen_ns:
            continue
        seen_ns.add(n)
        search_candidates.append((n, factors, f"high_div_{div_count}"))
        if len(search_candidates) >= 20:
            break
            
    candidates.extend(search_candidates)
    
    # 3. Evaluate tau_perp for all candidates
    best_score = 0
    best_n = candidates[0][0] if candidates else 2
    
    for n, factors, desc in candidates:
        if time.time() - start_time > DEADLINE - 2:
            break
            
        try:
            score = tau_perp_from_factors(factors)
            if score > best_score:
                best_score = score
                best_n = n
        except Exception as e:
            pass # Ignore invalid candidates
            
    return best_n
# EVOLVE-BLOCK-END
