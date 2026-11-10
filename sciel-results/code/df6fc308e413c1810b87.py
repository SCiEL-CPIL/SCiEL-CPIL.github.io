"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    answers = [None] * 254  # for n=3..256
    
    # Precompute primes up to 1000 for factorization
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    
    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors
    
    def get_divisors(factors):
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divisors = new_divs
        return divisors
    
    # Cache for factorization and divisors
    factor_cache = {}
    divisor_cache = {}
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # Get divisors of B^2
            if B not in factor_cache:
                factor_cache[B] = factorize(B)
            factors_B = factor_cache[B]
            factors_B2 = {p: 2*e for p, e in factors_B.items()}
            
            if B not in divisor_cache:
                divisor_cache[B] = get_divisors(factors_B2)
            divs = divisor_cache[B]
            
            # We need d such that:
            # 1. d divides B^2
            # 2. d >= d_min = max(1, A*x - B)
            # 3. d <= B (so d <= e = B^2/d)
            # 4. (d + B) % A == 0
            # 5. (B^2/d + B) % A == 0
            
            d_min = max(1, A * x - B)
            d_max = B
            
            if d_min > d_max:
                continue
            
            found_for_x = None
            for d in divs:
                if d < d_min or d > d_max:
                    continue
                if (d + B) % A != 0:
                    continue
                e = B * B // d
                if (e + B) % A != 0:
                    continue
                y = (d + B) // A
                z = (e + B) // A
                
                if y < x or z < y:
                    continue
                if x == y or y == z or x == z:
                    continue
                if z > 10**12:
                    continue
                
                found_for_x = [x, y, z]
                break
            
            if found_for_x:
                found = found_for_x
                break
        
        answers[idx] = found
    
    return answers
# EVOLVE-BLOCK-END
