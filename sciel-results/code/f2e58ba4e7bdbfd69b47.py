import time
from math import gcd
from functools import lru_cache

# Precompute primes up to a reasonable limit for factorization
def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

PRIMES = _sieve(1000)

@lru_cache(maxsize=None)
def _factorize(n):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    if n <= 1:
        return factors
    temp_n = n
    for p in PRIMES:
        if p * p > temp_n:
            break
        while temp_n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp_n //= p
    if temp_n > 1:
        factors[temp_n] = factors.get(temp_n, 0) + 1
    return factors

def _all_divisors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        powers = [p ** i for i in range(e + 1)]
        divs = [d * pw for d in divs for pw in powers]
    return divs

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    BOUND = 10**12
    
    answers = [None] * 254  # For n=3..256
    
    # First pass: quick search with bounded x,y,z for all n
    # Try x from n//4+1 to min(2*n, 1000)
    for n in range(3, 257):
        if time.time() - start_time > DEADLINE:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = min(2 * n, 1000)
        
        for x in range(x_min, x_max + 1):
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            # Now find y such that 1/y + 1/z = (4n - 4x)/(nx) = a/(nx) * ... 
            # Actually: 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx) = a/b
            # So (y+z)/(yz) = a/b => b(y+z) = a*yz => a*yz - b*y - b*z = 0
            # (ay-b)(az-b) = b^2
            b_sq = b * b
            bf = _factorize(b)
            b_sq_factors = {p: 2 * e for p, e in bf.items()}
            divs = _all_divisors(b_sq_factors)
            
            for d in divs:
                if d > b:
                    continue
                
                num_y = d + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                
                v = b_sq // d
                num_z = v + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                
                if x == y or x == z or y == z:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        if found:
            answers[n - 3] = found
    
    # Second pass: for unsolved n, do extended search
    for idx in range(254):
        n = idx + 3
        if answers[idx] is not None:
            continue
        
        if time.time() - start_time > DEADLINE:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = 5 * n
        
        for x in range(x_min, x_max + 1):
            if time.time() - start_time > DEADLINE:
                break
            
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            b_sq = b * b
            bf = _factorize(b)
            b_sq_factors = {p: 2 * e for p, e in bf.items()}
            divs = _all_divisors(b_sq_factors)
            
            for d in divs:
                if time.time() - start_time > DEADLINE:
                    break
                
                if d > b:
                    continue
                
                num_y = d + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                
                v = b_sq // d
                num_z = v + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                
                if x == y or x == z or y == z:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
