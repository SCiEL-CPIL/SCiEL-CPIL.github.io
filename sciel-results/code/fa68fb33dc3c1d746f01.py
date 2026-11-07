import math

def construct():
    n = 256
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    result = [1]
    for p in primes:
        # Find largest power of p <= n
        val = p
        while val * p <= n:
            val *= p
        result.append(val)
        
    return result
