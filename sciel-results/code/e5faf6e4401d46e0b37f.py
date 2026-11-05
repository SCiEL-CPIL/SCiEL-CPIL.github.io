"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin, though this solution is O(1) effectively
    
    # The optimal set consists of 1 and the largest prime power p^k <= 511 for each prime p <= 511.
    # Reasoning:
    # 1. Any composite number with >= 2 distinct prime factors blocks multiple prime powers.
    #    The sum of the weights of the blocked prime powers is strictly greater than the weight of the composite.
    # 2. For a prime power p^k, it is always better to choose the largest such power p^K <= 511,
    #    because it has a higher weight (smaller denominator) and blocks the same set of numbers (multiples of p).
    # 3. 1 is coprime to all and adds positive value, so it is always included.
    # 4. The set of distinct prime powers (one per prime) are pairwise coprime.
    
    N = 511
    result = [1]
    
    # Sieve of Eratosthenes to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    for p in range(2, N + 1):
        if is_prime[p]:
            # Find largest power p^k <= N
            power = p
            while power * p <= N:
                power *= p
            result.append(power)
            
    return result
# EVOLVE-BLOCK-END
