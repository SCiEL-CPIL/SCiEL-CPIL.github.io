import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s wall-clock
    
    # Precompute smallest prime factor (SPF) for fast factorization
    MAX_V = 1000000
    spf = list(range(MAX_V + 1))
    for i in range(2, int(MAX_V**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_V + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize_fast(n):
        """Factorize n using precomputed SPF."""
        factors = {}
        while n > 1:
            p = spf[n] if n <= MAX_V else n
            if p > MAX_V:
                # n is prime (or has a large prime factor)
                factors[p] = factors.get(p, 0) + 1
                n //= p
                continue
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = factors.get(p, 0) + cnt
        return factors
    
    def divisors_from_factors(factors):
        """Generate all divisors from prime factorization dict."""
        if not factors:
            return [1]
        primes = list(factors.keys())
        exps = [factors[p] for p in primes]
        divs = [1]
        for i, p in enumerate(primes):
            new_divs = []
            mult = 1
            for e in range(exps[i] + 1):
                for d in divs:
                    new_divs.append(d * mult)
                mult *= p
            divs = new_divs
        return divs
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x range: n/4 < x < 3n/4
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        # Precompute factorization of n
        n_factors = factorize_fast(n)
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            
            K = 4 * x - n
            if K <= 0:
                continue
            
            target = n * x
            target_sq = target * target
            
            # Factorize target_sq = (n*x)^2
            # Get factors of n*x first, then double exponents
            nx = n * x
            nx_factors = factorize_fast(nx)
            target_sq_factors = {}
            for p, e in nx_factors.items():
                target_sq_factors[p] = e * 2
            
            divisors = divisors_from_factors(target_sq_factors)
            
            for A in divisors:
                B = target_sq // A
                if (A + target) % K == 0:
                    if (B + target) % K == 0:
                        y = (A + target) // K
                        z = (B + target) // K
                        # Ensure distinctness: x, y, z all distinct and positive
                        if x != y and x != z and y != z:
                            if x > 0 and y > 0 and z > 0:
                                if z <= 10**12 and y <= 10**12:
                                    found = [x, y, z]
                                    break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
