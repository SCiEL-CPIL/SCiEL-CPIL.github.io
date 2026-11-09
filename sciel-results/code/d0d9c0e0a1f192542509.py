"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import numpy as np

def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = False
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    primes = _primes(N)
    # Start with all integers [0, N]
    S = set(range(N + 1))
    
    # Process primes in decreasing order
    for p in reversed(primes):
        if time.time() > deadline:
            break
        if p > len(S):
            continue
        # Count elements in each residue class mod p
        counts = np.zeros(p, dtype=int)
        for v in S:
            counts[v % p] += 1
        # Check if all classes are covered
        if np.all(counts > 0):
            # Remove the class with the fewest elements
            min_class = np.argmin(counts)
            S = {v for v in S if v % p != min_class}
    
    return list(S)
# EVOLVE-BLOCK-END
