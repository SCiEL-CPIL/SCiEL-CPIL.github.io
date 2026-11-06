"""Rewritten program: coprime-moduli greedy for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import math


def _all_primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def _coprime_friendly_moduli(lo=20, hi=200):
    """All primes and prime powers in [lo, hi]; any two have distinct prime bases => coprime."""
    primes = _all_primes_up_to(hi)
    moduli = set()
    for p in primes:
        v = p
        while v < lo:
            v *= p
        while v <= hi:
            moduli.add(v)
            v *= p
    return sorted(moduli)


def construct():
    # Candidate moduli: primes + prime powers in [20, 200] (mutually coprime).
    candidates = _coprime_friendly_moduli(20, 200)

    chosen = []  # list of [m, r]
    for m in candidates:  # increasing m => decreasing 1/m, greedy by value
        # find a residue r in [0, m) disjoint from all chosen classes
        placed = False
        for r in range(m):
            ok = True
            for (m2, r2) in chosen:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                chosen.append([m, r])
                placed = True
                break
        # if no residue works (shouldn't happen for coprime set), skip m
    return chosen
# EVOLVE-BLOCK-END
