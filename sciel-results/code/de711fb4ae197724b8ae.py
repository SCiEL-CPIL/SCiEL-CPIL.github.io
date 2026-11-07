"""Search for M with many distinct k satisfying k*sigma(k)=M, k<=sqrt(M)."""
import math
import time

DEADLINE = time.time() + 140.0
MAX_K = 5_000_000
M_CAP = 10**18


def _factor_sigma(n):
    # return sigma(n) via trial division
    res = 1
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            s = 1
            pk = 1
            while m % p == 0:
                m //= p
                pk *= p
                s += pk
            res *= s
        p += 1 if p == 2 else 2
    if m > 1:
        res *= (1 + m)
    return res


def construct():
    # Precompute primes up to sqrt(MAX_K) for fast sigma
    limit = int(math.isqrt(MAX_K)) + 1
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, limit + 1) if sieve[i]]

    from collections import defaultdict
    best_groups = {}
    best_M = 0
    best_k = []

    for k in range(1, MAX_K + 1):
        if time.time() > DEADLINE:
            break
        # sigma via prime list
        m = k
        s = 1
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                term = 1
                pk = 1
                while m % p == 0:
                    m //= p
                    pk *= p
                    term += pk
                s *= term
        if m > 1:
            s *= (1 + m)
        M = k * s
        if M > M_CAP:
            continue
        g = best_groups.get(M)
        if g is None:
            best_groups[M] = [k]
        else:
            g.append(k)
            if len(g) > len(best_k):
                best_k = g[:]
                best_M = M

    if best_k:
        # filter k <= sqrt(M) and verify
        root = int(math.isqrt(best_M))
        ks = [k for k in best_k if k <= root]
        # dedupe + sort
        ks = sorted(set(ks))
        # verify
        ok = []
        for k in ks:
            if k * _factor_sigma(k) == best_M:
                ok.append(k)
        if ok:
            return {'M': int(best_M), 'k': [int(x) for x in ok]}

    # fallback known
    return {'M': 333312, 'k': [336, 372, 434]}
