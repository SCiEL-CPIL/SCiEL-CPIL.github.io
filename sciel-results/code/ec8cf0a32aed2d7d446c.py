import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute smallest prime factor up to max possible x
    # max n = 256, max x = 3*256//4 = 192
    # B = n*x, max B = 256*192 = 49152
    # We need factors of B. B is product of n and x.
    # n <= 256, x <= 192.
    # We can just factor B directly using a simple trial division up to sqrt(B) or precomputed primes.
    # Primes up to sqrt(49152) ~ 221.
    # Let's generate primes up to 300.
    
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        primes = [i for i, prime in enumerate(is_prime) if prime]
        return primes

    primes = sieve(300)
    
    def factorize(num, primes):
        factors = {}
        for p in primes:
            if p * p > num:
                break
            while num % p == 0:
                factors[p] = factors.get(p, 0) + 1
                num //= p
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors

    def get_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                for i in range(e + 1):
                    new_divs.append(d * (p ** i))
            divs = new_divs
        return divs

    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
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
            
            B_sq = B * B
            d_min = max(1, A * x - B)
            d_max = B
            
            if d_min > d_max:
                continue
            
            # Factor B
            factors_B = factorize(B, primes)
            # Factors of B^2 have exponents 2*e
            factors_B_sq = {p: 2*e for p, e in factors_B.items()}
            
            # Get divisors of B^2
            divs = get_divisors(factors_B_sq)
            
            # Filter and check
            found_for_x = None
            for d in divs:
                if d < d_min or d > d_max:
                    continue
                
                if (d + B) % A != 0:
                    continue
                
                e = B_sq // d
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
        
        answers.append(found)
    
    return answers
