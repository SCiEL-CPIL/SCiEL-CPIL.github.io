"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import reduce
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    LIMIT = 10**18
    
    # First 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def get_divisors(prime_list):
        """Generate all divisors of product of primes (squarefree)."""
        divs = [1]
        for p in prime_list:
            divs = [d * p for d in divs] + divs
        return sorted(divs)
    
    def tau_perp(n, prime_list):
        """Count adjacent coprime pairs in sorted divisors."""
        divs = get_divisors(prime_list)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    # Evaluate the base: product of first 15 primes
    base_n = reduce(lambda x, y: x * y, primes)
    assert base_n <= LIMIT, f"Base exceeds limit: {base_n}"
    
    best_score = tau_perp(base_n, primes)
    best_n = base_n
    
    # Try: what if we use fewer primes but with some higher exponents?
    # For squarefree, more primes = more divisors = more potential coprime pairs
    # But let's also try: product of first 14 primes * some small power
    # e.g., 2^2 * 3 * 5 * ... * 43 (replace 47 with 2*47? No, that changes structure)
    
    # Try: first 14 primes, multiply by 49 (7^2) instead of 47
    # n = 2*3*5*7^2*11*13*17*19*23*29*31*37*41*43
    # This has more divisors: 2^13 * 3 = 24576... wait, 14 primes with one exponent 2
    # Actually: 13 primes with exp 1, 1 prime with exp 2: 2^13 * 3 = 24576 divisors
    # vs 15 primes all exp 1: 2^15 = 32768 divisors
    # So 15 primes is better in count.
    
    # Try: can we fit 15 primes plus multiply by a small factor?
    # base_n * 2 = 1229779565176982820 > 10^18? No: 614889782588491410 * 2 = 1229779565176982820 < 10^18
    # But that makes 2^2 * 3 * 5 * ... * 47, which has 2^14 * 2 = 32768 divisors (same count)
    # But the structure changes - 2 appears with exponent 2.
    
    # Let's try a few variations
    variations = []
    
    # Variation 1: first 15 primes (baseline)
    variations.append((primes[:15], [1]*15))
    
    # Variation 2: first 14 primes, 47 replaced by 49=7^2 (but 7 already there, so 7^3)
    # Actually let's just try: 2^2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # That's 2^2 and others exp 1: 2^14 * 2 = 32768 divisors
    p14 = primes[:14]  # 2..43
    n_var2 = 2 * reduce(lambda x, y: x * y, p14)  # 2^2 * 3*5*...*43
    if n_var2 <= LIMIT:
        # Divisors: for 2^2*3*5*...*43, count = 3 * 2^13 = 24576
        # Fewer divisors than 32768, likely worse
        pass
    
    # The key question: does squarefree with 15 primes maximize tau_perp?
    # Let's also try: what about using prime 53 instead of 47? No, 47 is smaller.
    
    # Try: first 13 primes * 47 * 53? 
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
    # * 53 = too big. 
    
    # Try replacing 47 with a composite? No, must be prime for squarefree.
    
    # Let's try: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 is our best candidate.
    # Score it and return.
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    return best_n
# EVOLVE-BLOCK-END
