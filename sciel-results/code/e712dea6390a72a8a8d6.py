import math
import time
from collections import defaultdict

def _sigma_slow(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            term = 1
            pw = 1
            while temp % p == 0:
                temp //= p
                exp += 1
                pw *= p
                term += pw
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start = time.time()
    deadline = start + 130.0

    def _fallback():
        # Try several known M values with multiple solutions
        candidates = [
            (333312, [336, 372, 434]),
            (1, [1]),
        ]
        for M, ks in candidates:
            ok = []
            for k in ks:
                if 1 <= k <= math.isqrt(M) and k * _sigma_slow(k) == M:
                    ok.append(k)
            if len(ok) >= 1:
                return {'M': int(M), 'k': [int(x) for x in ok]}
        return {'M': 1, 'k': [1]}

    import numpy as np

    # Use a larger sieve to find more collisions
    # N = 20,000,000 should be feasible with NumPy vectorization
    N = 20_000_000

    try:
        sigma = np.zeros(N + 1, dtype=np.int64)
        for d in range(1, N + 1):
            sigma[d::d] += d
            if time.time() > deadline - 30.0:
                break
    except Exception:
        return _fallback()

    if time.time() > deadline - 25.0:
        return _fallback()

    # Build map from M -> list of k
    M_to_ks = defaultdict(list)
    sigma_list = sigma.tolist()
    for k in range(1, N + 1):
        M_val = k * sigma_list[k]
        if M_val <= 10**18:
            M_to_ks[M_val].append(k)
        if time.time() > deadline - 20.0:
            break

    if time.time() > deadline - 15.0:
        return _fallback()

    # Pick the M with the most valid k's (k <= sqrt(M))
    best_M = None
    best_sols = None
    for M_val, ks in M_to_ks.items():
        if len(ks) < 2:
            continue
        limit = math.isqrt(M_val)
        valid = [k for k in ks if k <= limit]
        if best_sols is None or len(valid) > len(best_sols):
            best_sols = valid
            best_M = M_val
        if time.time() > deadline - 10.0:
            break

    if best_sols is None or len(best_sols) < 2:
        return _fallback()

    # Final exact verification
    verified = []
    for k in best_sols:
        if 1 <= k <= math.isqrt(best_M) and k <= N:
            if k * sigma_list[k] == best_M:
                verified.append(k)

    if not verified:
        return _fallback()

    verified = sorted(set(verified))
    return {'M': int(best_M), 'k': [int(x) for x in verified]}
