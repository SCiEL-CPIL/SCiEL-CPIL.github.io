import math

def construct():
    # Limit for primes is sqrt(max_n)
    MAX_N = 1000000
    LIMIT = 1000
    
    # Sieve of Eratosthenes up to LIMIT
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def powerful_part(n):
        p_part = 1
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                power = 1
                while temp % p == 0:
                    temp //= p
                    power += 1
                if power >= 2:
                    p_part *= (p ** power)
        # If temp > 1, it is a prime factor with exponent 1, so it is not included in powerful part
        return p_part

    best_n = 10000
    best_val = -1
    
    for n in range(10000, MAX_N):
        pn = powerful_part(n)
        pn1 = powerful_part(n + 1)
        val = pn * pn1
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
