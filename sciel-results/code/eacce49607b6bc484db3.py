"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 148.0  # Safety margin before 150s limit
    
    LIMIT = 10**18
    
    # First 16 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
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
    
    best_n = 1
    best_score = 0
    
    # Strategy: Try different numbers of distinct primes (k) and distribute
    # extra budget as exponents on the smallest primes.
    
    # For each k from 16 down to 10, compute base product and remaining budget
    for k in range(16, 9, -1):
        if time.time() - start_time > DEADLINE:
            break
        prime_subset = primes[:k]
        base_product = 1
        for p in prime_subset:
            base_product *= p
        
        if base_product > LIMIT:
            continue
        
        remaining = LIMIT // base_product
        
        # Try various exponent distributions on the first few primes
        # Extra exponent on prime i means we multiply by prime_i^(extra_i)
        # We want to try: put all extra on smallest, split between first two, etc.
        
        # Case 1: All extra on prime 2 (2^e where 2^e <= remaining)
        max_e2 = 0
        temp = remaining
        while temp >= 2:
            temp //= 2
            max_e2 += 1
        
        for e2 in range(max_e2, -1, -1):
            if time.time() - start_time > DEADLINE:
                break
            factors = [(2, 1 + e2)] + [(p, 1) for p in prime_subset[1:]]
            n_val = base_product * (2 ** e2)
            if n_val > LIMIT:
                continue
            score = tau_perp_from_factors(factors)
            if score > best_score:
                best_score = score
                best_n = n_val
        
        # Case 2: Split extra between 2 and 3
        max_e2 = 0
        temp = remaining
        while temp >= 2:
            temp //= 2
            max_e2 += 1
        
        for e2 in range(min(max_e2, 8), -1, -1):
            if time.time() - start_time > DEADLINE:
                break
            rem_after_2 = remaining // (2 ** e2)
            if rem_after_2 < 1:
                continue
            max_e3 = 0
            temp = rem_after_2
            while temp >= 3:
                temp //= 3
                max_e3 += 1
            
            for e3 in range(min(max_e3, 6), -1, -1):
                if time.time() - start_time > DEADLINE:
                    break
                factors = [(2, 1 + e2), (3, 1 + e3)] + [(p, 1) for p in prime_subset[2:]]
                n_val = base_product * (2 ** e2) * (3 ** e3)
                if n_val > LIMIT:
                    continue
                score = tau_perp_from_factors(factors)
                if score > best_score:
                    best_score = score
                    best_n = n_val
        
        # Case 3: Split extra among 2, 3, 5
        max_e2 = 0
        temp = remaining
        while temp >= 2:
            temp //= 2
            max_e2 += 1
        
        for e2 in range(min(max_e2, 5), -1, -1):
            if time.time() - start_time > DEADLINE:
                break
            rem_after_2 = remaining // (2 ** e2)
            if rem_after_2 < 1:
                continue
            max_e3 = 0
            temp = rem_after_2
            while temp >= 3:
                temp //= 3
                max_e3 += 1
            
            for e3 in range(min(max_e3, 4), -1, -1):
                if time.time() - start_time > DEADLINE:
                    break
                rem_after_3 = rem_after_2 // (3 ** e3)
                if rem_after_3 < 1:
                    continue
                max_e5 = 0
                temp = rem_after_3
                while temp >= 5:
                    temp //= 5
                    max_e5 += 1
                
                for e5 in range(min(max_e5, 3), -1, -1):
                    if time.time() - start_time > DEADLINE:
                        break
                    factors = [(2, 1 + e2), (3, 1 + e3), (5, 1 + e5)] + [(p, 1) for p in prime_subset[3:]]
                    n_val = base_product * (2 ** e2) * (3 ** e3) * (5 ** e5)
                    if n_val > LIMIT:
                        continue
                    score = tau_perp_from_factors(factors)
                    if score > best_score:
                        best_score = score
                        best_n = n_val
    
    if best_n < 2:
        best_n = 2
    
    return best_n
# EVOLVE-BLOCK-END
