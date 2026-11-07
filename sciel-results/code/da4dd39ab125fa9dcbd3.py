"""
Rewritten program using a CRT constructive approach combined with LPF sieve.
For each gap (lo, hi) with g = hi - lo, we need m in (lo, hi) with lpf(m) >= g.
If m is prime, lpf(m) = m >= g (since m > lo >= 10000 >> g for these gaps).
So any prime in the gap works. For composite m, we need all prime factors >= g.

Strategy:
1. Use LPF sieve for O(1) lookup of least prime factor.
2. For each gap, scan for the first m in (lo, hi) with lpf[m] >= g.
3. Additionally, use CRT to construct candidates: m = k*M + 1 where M = product of primes < g.
   Such m is not divisible by any prime < g, so lpf(m) >= g (unless m < g, which won't happen).
   We check if any such m falls in (lo, hi).
"""

import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_lpf(n):
    """Compute least prime factor array up to n."""
    lpf = [0] * (n + 1)
    for i in range(2, n + 1):
        if lpf[i] == 0:
            for j in range(i, n + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    return lpf


def _primes_below(g):
    """Return list of primes less than g."""
    if g <= 2:
        return []
    # Simple sieve up to g-1
    is_prime = [True] * g
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(g ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, g, i):
                is_prime[j] = False
    return [i for i in range(2, g) if is_prime[i]]


def _crt_candidates(lo, hi, g, max_k=10000):
    """
    Generate candidates m = k*M + 1 where M = product of primes < g.
    These are not divisible by any prime < g, so lpf(m) >= g (for m >= g).
    Return list of such m in (lo, hi).
    """
    if g <= 2:
        # M = 1, so m = k*1 + 1 = k+1; any m works if lpf(m) >= 2 (i.e., m is odd or prime)
        # Actually for g=2, we just need lpf(m) >= 2, which is always true for m >= 2.
        # So any integer in (lo, hi) works.
        return []
    
    primes = _primes_below(g)
    if not primes:
        return []
    
    # Compute M = product of primes < g
    # But M might be very large. For g up to ~20, M is manageable.
    # For larger g, M grows fast. Let's cap it.
    M = 1
    for p in primes:
        M *= p
        if M > 10**12:
            # M too large; k*M + 1 will skip over (lo, hi) quickly
            # Only k=0 gives m=1 which is not in range
            break
    
    candidates = []
    # m = k*M + 1 must be in (lo, hi)
    # k*M + 1 > lo  =>  k > (lo - 1) / M
    # k*M + 1 < hi  =>  k < (hi - 1) / M
    k_min = (lo - 1) // M + 1
    k_max = (hi - 1) // M - 1
    
    if k_min > k_max:
        # Try k starting from 0 up to some limit
        # Actually k_min might be 0 if lo < 1, but lo >= 10000
        # Let's just iterate k from max(0, k_min) to min(k_max, max_k)
        k_start = max(0, k_min)
        k_end = min(k_max, max_k)
        if k_start > k_end:
            return []
        for k in range(k_start, k_end + 1):
            m = k * M + 1
            if lo < m < hi:
                candidates.append(m)
    else:
        for k in range(k_min, min(k_max, max_k) + 1):
            m = k * M + 1
            if lo < m < hi:
                candidates.append(m)
    
    return candidates


def construct():
    """
    Find witnesses m in (lo, hi) such that lpf(m) >= gap = hi - lo.
    Uses LPF sieve for O(1) lookup, plus CRT construction as backup.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline for safety margin
    
    # Determine the maximum value we need to check
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Sieve for Least Prime Factor (LPF)
    lpf = _sieve_lpf(max_val)
    
    if time.time() > deadline:
        return [None] * len(P['gaps'])
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        witness = None
        
        # First: scan for any m in (lo, hi) with lpf[m] >= gap
        # This includes primes (lpf[p] = p >= gap since p > lo >> gap)
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                witness = m
                break
        
        if witness is None:
            # Try CRT construction
            candidates = _crt_candidates(lo, hi, gap)
            for m in candidates:
                # Verify lpf[m] >= gap
                if m <= max_val and lpf[m] >= gap:
                    witness = m
                    break
                elif m > max_val:
                    # Need to compute lpf for m; use trial division up to gap
                    # m is not divisible by any prime < gap by construction
                    # So lpf[m] >= gap is guaranteed if m >= gap
                    if m >= gap:
                        witness = m
                        break
        
        result[i] = witness
    
    # Ensure types are strictly Python int or None
    return [w if isinstance(w, int) and not isinstance(w, bool) else None for w in result]
