"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 2s margin
    
    n = 256
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    result = [1]
    for p in primes:
        # Find largest power of p <= n
        max_power = p
        current = p
        while current * p <= n:
            current *= p
            max_power = current
        result.append(max_power)
        
    # Basic validation
    # 1. Check distinct
    if len(result) != len(set(result)):
        return [1] # Fallback, should not happen
    # 2. Check range
    if any(x < 1 or x > 256 for x in result):
        return [1]
    # 3. Check pairwise coprime
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                return [1]
                
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
