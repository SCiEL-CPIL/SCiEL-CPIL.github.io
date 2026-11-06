"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _primes_up_to(n):
    """Return list of primes from 2 to n inclusive."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    """
    Return a list of [residue, modulus] pairs with distinct moduli in [2, 128],
    canonical residues 0 <= r < m, such that every two infinite residue classes
    are disjoint. Maximize the number of classes.
    
    Strategy: Use all primes from 2 to 128 as moduli. Since any two distinct primes
    are coprime (gcd = 1), any two residue classes with coprime moduli are
    automatically disjoint (the CRT compatibility condition a ≡ b (mod gcd(m,n))
    becomes a ≡ b (mod 1), which is always satisfied, meaning the classes are
    disjoint for ANY choice of residues). We use residue 0 for all.
    """
    primes = _primes_up_to(128)
    # Each prime p gives class 0 mod p.
    # All pairs are disjoint because gcd(p, q) = 1 for distinct primes p, q.
    return [[0, p] for p in primes]
# EVOLVE-BLOCK-END
