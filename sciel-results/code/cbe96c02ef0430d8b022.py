import math
import time
import sys

def _sigma(n):
    """Exact sigma(n) via trial division (with isqrt cutoff)."""
    if n <= 0:
        return 0
    s = 1
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            term = 1
            pk = 1
            while x % p == 0:
                x //= p
                pk *= p
                term += pk
            s *= term
        p += 1 if p == 2 else 2
    if x > 1:
        s *= (1 + x)
    return s

def _sigma_sieve(nmax):
    """Compute sigma(k) for k=1..nmax via sieve. Returns list length nmax+1."""
    sigma = [0] * (nmax + 1)
    for i in range(1, nmax + 1):
        for j in range(i, nmax + 1, i):
            sigma[j] += i
    return sigma

def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    # Fallback: known 3-way collision
    best = {'M': 333312, 'k': [336, 372, 434]}

    # Try progressively larger sieve sizes to find bigger collisions
    # Each pass: compute k*sigma(k) for k in [1, N], group by value, find max group
    # among groups where all k <= isqrt(M) and M <= 10^18.
    sizes = [100000, 300000, 800000, 1500000, 3000000, 6000000, 12000000, 25000000]

    for N in sizes:
        if time.time() - t0 > DEADLINE:
            break
        try:
            sigma = _sigma_sieve(N)
            # Build dict: value -> list of k (only k where k*sigma(k) <= 10^18 and k <= isqrt(k*sigma(k)))
            from collections import defaultdict
            groups = defaultdict(list)
            for k in range(1, N + 1):
                M = k * sigma[k]
                if M > 10**18:
                    continue
                # check k <= isqrt(M)
                if k > math.isqrt(M):
                    continue
                groups[M].append(k)

            # Find best group
            best_group = None
            best_len = len(best['k'])
            for M, ks in groups.items():
                if len(ks) > best_len:
                    best_len = len(ks)
                    best_group = (M, ks)
            if best_group is not None:
                M, ks = best_group
                # Verify each with exact sigma
                verified = []
                for k in ks:
                    if k * _sigma(k) == M and k <= math.isqrt(M):
                        verified.append(k)
                if len(verified) > len(best['k']):
                    best = {'M': M, 'k': verified}
        except Exception:
            break

    # Ensure types
    M = int(best['M'])
    ks = [int(x) for x in best['k']]
    # Verify once more for safety
    final_ks = []
    for k in ks:
        if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= math.isqrt(M) and k * _sigma(k) == M:
            final_ks.append(k)
    # Deduplicate preserving order
    seen = set()
    dedup = []
    for k in final_ks:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    if len(dedup) == 0:
        # ultimate fallback
        return {'M': 333312, 'k': [336, 372, 434]}
    return {'M': M, 'k': dedup}
