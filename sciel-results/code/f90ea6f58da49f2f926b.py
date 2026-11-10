"""Search for M <= 1e18 with many k such that k*sigma(k)=M, using SPF-based sigma."""
import math
import time
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    # ---- SPF sieve up to N ----
    N = 5000000
    spf = list(range(N + 1))
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # ---- compute sigma(k) using SPF with prime-power tracking ----
    sigma = [0] * (N + 1)
    sigma[1] = 1
    # pp[n] = highest power of spf[n] dividing n
    pp = [1] * (N + 1)
    # r[n] = n / pp[n] (the part coprime to spf[n])
    r = [1] * (N + 1)
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m, so extend the prime power
            pp[n] = pp[m] * p
            r[n] = r[m]
        else:
            # p does not divide m, so pp[n] = p, r[n] = m
            pp[n] = p
            r[n] = m
        # sigma(n) = sigma(r[n]) * (p^(a+1)-1)/(p-1) where p^a = pp[n]
        # But we need to compute (p^(a+1)-1)/(p-1) efficiently
        # Note: pp[n] = p^a, so (p^(a+1)-1)/(p-1) = (pp[n]*p - 1)/(p-1)
        a_power = pp[n]  # p^a
        s_prime_power = (a_power * p - 1) // (p - 1)  # 1 + p + ... + p^a
        sigma[n] = sigma[r[n]] * s_prime_power

    # ---- group by M ----
    groups = defaultdict(list)
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            groups[M].append(k)

    # ---- find best ----
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    for M, ks in groups.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
            if best_count >= 1000:
                break

    # ---- verify ----
    def factorize(n):
        out = {}
        d = 2
        while d * d <= n:
            if n % d == 0:
                c = 0
                while n % d == 0:
                    n //= d
                    c += 1
                out[d] = c
            d += 1 if d == 2 else 2
        if n > 1:
            out[n] = 1
        return out

    def sigma_exact(k):
        f = factorize(k)
        s = 1
        for p, e in f.items():
            s *= (p ** (e + 1) - 1) // (p - 1)
        return s

    if time.time() < DEADLINE - 3.0:
        verified = []
        for k in best_k:
            if k > math.isqrt(best_M):
                continue
            if k * sigma_exact(k) == best_M:
                verified.append(k)
        if verified:
            best_k = verified
    else:
        best_k = [k for k in best_k if k <= math.isqrt(best_M)]

    best_k = sorted(set(best_k))[:1000]
    if not best_k:
        return {'M': 333312, 'k': [336, 372, 434]}
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
