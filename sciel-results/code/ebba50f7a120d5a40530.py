"""Solve max pairwise-coprime subset of 1..256 by taking 1 and the largest prime power <= 256 for each prime."""
# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _max_prime_power(p, limit):
    """Largest power of prime p that is <= limit."""
    val = p
    while val * p <= limit:
        val *= p
    return val

def construct():
    limit = 256
    primes = _primes_up_to(limit)
    result = [1]
    for p in primes:
        result.append(_max_prime_power(p, limit))
    return result

# EVOLVE-BLOCK-END
