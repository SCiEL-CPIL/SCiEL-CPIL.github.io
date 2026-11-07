"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    t0 = time.time()
    deadline = t0 + 80.0

    N = 256
    # Sieve primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [x for x in range(2, N + 1) if is_prime[x]]

    def distinct_prime_factors(n):
        pf = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                pf.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            pf.add(n)
        return pf

    # Start with all primes
    S = set(primes)

    # Add composites that are safe with the full prime set present
    for m in range(4, N + 1):
        if is_prime[m]:
            continue
        pf = distinct_prime_factors(m)
        if len(pf) >= 3:
            S.add(m)
        elif len(pf) == 1:
            # Prime power p^k, k>=2: always safe
            S.add(m)
        # len(pf)==2: skip for now, will try local search

    # Local search: try to add semiprimes (2 distinct prime factors) by removing one prime
    # For each semiprime m = p*q not in S, try removing p or q
    semiprimes = []
    for m in range(4, N + 1):
        if is_prime[m]:
            continue
        pf = distinct_prime_factors(m)
        if len(pf) == 2:
            semiprimes.append(m)

    semiprimes.sort()

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for m in semiprimes:
            if time.time() > deadline:
                break
            if m in S:
                continue
            pf = distinct_prime_factors(m)
            p1, p2 = sorted(pf)
            for p_to_remove in [p1, p2]:
                if p_to_remove not in S:
                    continue
                # Trial: remove p_to_remove, add m
                # Check m against all pairs in (S - {p_to_remove})
                ok = True
                elems = [x for x in S if x != p_to_remove and x != m]
                for i in range(len(elems)):
                    for j in range(i + 1, len(elems)):
                        if (elems[i] * elems[j]) % m == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    S.discard(p_to_remove)
                    S.add(m)
                    improved = True
                    break

    result = sorted(S)
    return result
# EVOLVE-BLOCK-END
