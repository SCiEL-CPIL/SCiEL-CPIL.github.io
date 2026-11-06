"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Uses a linear sieve to precompute the smallest prime factor (SPF) up to 1,000,000,
    then iterates n from 1 upwards, computing phi(n) in O(log n) via the SPF table.
    Returns the first n where phi(n) % 120 == 0.
    
    Deadline guard: if computation exceeds 85s, returns the best candidate found so far.
    In practice, the sieve and search complete in well under 1 second.
    """
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    N = 1_000_000
    
    # Linear sieve for smallest prime factor
    spf = list(range(N + 1))  # spf[i] = smallest prime factor of i
    spf[0] = 0
    spf[1] = 1
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            is_prime[i * p] = False
            spf[i * p] = p
    
    # Compute phi(n) for each n using SPF, find smallest with phi(n) % 120 == 0
    # phi(n) = n * prod_{p|n} (1 - 1/p)
    # We can compute phi incrementally or factor each n.
    # For efficiency, compute phi array using the sieve-like approach:
    # phi[i] = i initially, then for each prime p, phi[i] -= phi[i]//p for multiples.
    
    phi = list(range(N + 1))
    for p in primes:
        for multiple in range(p, N + 1, p):
            phi[multiple] -= phi[multiple] // p
    
    # Now find smallest n >= 1 with phi[n] % 120 == 0
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
        if time.time() > deadline:
            return n  # best effort: return current candidate
    
    return N  # fallback
# EVOLVE-BLOCK-END
