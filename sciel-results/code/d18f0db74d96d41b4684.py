"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import reduce

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    LIMIT = 10**18
    
    # First 16 primes - check if product fits
    primes_16 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    primes_15 = primes_16[:15]
    
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
    
    # Try 16 primes
    n16 = 1
    for p in primes_16:
        n16 *= p
    
    best_n = 0
    best_score = 0
    
    if n16 <= LIMIT:
        score16 = tau_perp_from_factors([(p, 1) for p in primes_16])
        if score16 > best_score:
            best_score = score16
            best_n = n16
    
    # Try 15 primes
    n15 = 1
    for p in primes_15:
        n15 *= p
    
    if n15 <= LIMIT:
        score15 = tau_perp_from_factors([(p, 1) for p in primes_15])
        if score15 > best_score:
            best_score = score15
            best_n = n15
    
    # Try variations: 15 primes but with one prime squared to use remaining budget
    # n15 * 2 = 2^2 * 3 * 5 * ... * 47
    n_var = n15 * 2
    if n_var <= LIMIT:
        factors_var = [(2, 2)] + [(p, 1) for p in primes_15[1:]]
        score_var = tau_perp_from_factors(factors_var)
        if score_var > best_score:
            best_score = score_var
            best_n = n_var
    
    # Try: 14 primes * 2^3 (use more budget)
    primes_14 = primes_15[:14]
    n14 = 1
    for p in primes_14:
        n14 *= p
    # n14 * 8 = 2^3 * 3 * 5 * ... * 43
    n_var2 = n14 * 8
    if n_var2 <= LIMIT:
        factors_var2 = [(2, 3)] + [(p, 1) for p in primes_14[1:]]
        score_var2 = tau_perp_from_factors(factors_var2)
        if score_var2 > best_score:
            best_score = score_var2
            best_n = n_var2
    
    # Try: 14 primes * 2^2 * 3^2
    n_var3 = n14 * 4 * 3
    if n_var3 <= LIMIT:
        factors_var3 = [(2, 2), (3, 2)] + [(p, 1) for p in primes_14[2:]]
        score_var3 = tau_perp_from_factors(factors_var3)
        if score_var3 > best_score:
            best_score = score_var3
            best_n = n_var3
    
    # Try: 13 primes with various exponent distributions
    primes_13 = primes_15[:13]
    n13 = 1
    for p in primes_13:
        n13 *= p
    remaining = LIMIT // n13
    
    # Try 2^4 * 3 * 5 * ... * 41
    n_var4 = n13 * 8
    if n_var4 <= LIMIT:
        factors_var4 = [(2, 4)] + [(p, 1) for p in primes_13[1:]]
        score_var4 = tau_perp_from_factors(factors_var4)
        if score_var4 > best_score:
            best_score = score_var4
            best_n = n_var4
    
    # Try: 12 primes with more exponents
    primes_12 = primes_15[:12]
    n12 = 1
    for p in primes_12:
        n12 *= p
    remaining12 = LIMIT // n12
    
    # 2^5 * 3 * 5 * ... * 37
    n_var5 = n12 * 16
    if n_var5 <= LIMIT:
        factors_var5 = [(2, 5)] + [(p, 1) for p in primes_12[1:]]
        score_var5 = tau_perp_from_factors(factors_var5)
        if score_var5 > best_score:
            best_score = score_var5
            best_n = n_var5
    
    # 2^4 * 3^2 * 5 * ... * 37
    n_var6 = n12 * 8 * 3
    if n_var6 <= LIMIT:
        factors_var6 = [(2, 4), (3, 2)] + [(p, 1) for p in primes_12[2:]]
        score_var6 = tau_perp_from_factors(factors_var6)
        if score_var6 > best_score:
            best_score = score_var6
            best_n = n_var6
    
    # 2^3 * 3^3 * 5 * ... * 37
    n_var7 = n12 * 4 * 9
    if n_var7 <= LIMIT:
        factors_var7 = [(2, 3), (3, 3)] + [(p, 1) for p in primes_12[2:]]
        score_var7 = tau_perp_from_factors(factors_var7)
        if score_var7 > best_score:
            best_score = score_var7
            best_n = n_var7
    
    # 2^3 * 3^2 * 5^2 * 7 * ... * 37
    n_var8 = n12 * 4 * 3 * 5
    if n_var8 <= LIMIT:
        factors_var8 = [(2, 3), (3, 2), (5, 2)] + [(p, 1) for p in primes_12[3:]]
        score_var8 = tau_perp_from_factors(factors_var8)
        if score_var8 > best_score:
            best_score = score_var8
            best_n = n_var8
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    return best_n
# EVOLVE-BLOCK-END
