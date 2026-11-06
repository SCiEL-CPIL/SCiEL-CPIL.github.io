"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import reduce

def _sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    N = 512
    primes = _sieve_primes(N - 1)
    # Sort primes by 1/(N - p) descending, i.e., by p descending
    primes_desc = sorted(primes, reverse=True)
    
    chosen = []
    for p in primes_desc:
        chosen.append(p)
    
    # Try to add composite numbers that are coprime to all chosen
    # Candidates are numbers 1..511 not in chosen
    # We want to maximize sum 1/(N - a), so prefer larger a
    remaining = list(range(N - 1, 0, -1))
    remaining = [a for a in remaining if a not in chosen]
    
    for a in remaining:
        if all(math.gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    return chosen
# EVOLVE-BLOCK-END
