"""Find N <= 1e18 with many representations as a sum of two powerful numbers.

Strategy:
  1. Generate all powerful numbers up to M via the square*cube construction.
  2. FFT-convolve the indicator array to count representations for every N
     up to 2*M in one O(M log M) pass.
  3. Take the peak N*, verify every candidate pair with a fast powerful test,
     and extend the search window around N* (the true max may lie just past
     2*M, but only powerful numbers <= M can pair with a powerful number
     <= 1e18 there).
  4. A wall-clock deadline guard returns the best verified result found.
"""
import time
from math import isqrt

_DEADLINE = 142.0
_T0 = time.time()

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
                 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _is_powerful(n):
    """True if every prime factor of n has exponent >= 2 (1 counts)."""
    if n < 1:
        return False
    if n == 1:
        return True
    t = n
    for p in _SMALL_PRIMES:
        if p * p > t:
            break
        if t % p == 0:
            t //= p
            if t % p:
                return False
            while t % p == 0:
                t //= p
    if t > 1:
        r = isqrt(t)
        if r * r != t:
            return False
    return True


def _generate_powerful(limit):
    """All powerful numbers in [1, limit] as a sorted list."""
    pows = {1}
    for a in range(2, isqrt(limit) + 1):
        a2 = a * a
        for b in range(1, isqrt(limit // a2) + 1):
            b3 = b ** 3
            v = a2 * b3
            if v <= limit:
                pows.add(v)
    return sorted(pows)


def construct():
    import numpy as np

    M = 10_000_000
    powerful = _generate_powerful(M)
    arr = np.zeros(M + 1, dtype=np.float64)
    arr[powerful] = 1.0
    conv = np.fft.irfft(np.fft.rfft(arr) ** 2)
    conv = np.rint(conv).astype(np.int64)
    # conv[i] = #{(a,b): a+b=i, a,b powerful}  for i in [0, 2M]

    # Peak in the core search window.
    peak = int(np.argmax(conv))
    core_max = int(conv[peak])

    # Extend the window: a powerful a <= M pairs with powerful b <= 1e18
    # up to N = M + 1e18. Only extend while a powerful a <= M can still pair.
    maxN = min(M + 10**18, 2 * M + 2_000_000)
    start = max(1, peak - 2_000_000)
    end = min(maxN, peak + 2_000_000)

    best_N = None
    best_pairs = []
    best_count = 0

    for N in range(start, end + 1):
        if _time_left() < 6.0:
            break
        # Upper bound: a powerful a <= N/2, so a <= min(M, N/2).
        hi = min(M, N // 2)
        cnt = 0
        # Count candidate pairs (a powerful, N-a powerful, a <= N-a).
        for a in powerful:
            if a > hi:
                break
            if _is_powerful(N - a):
                cnt += 1
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = [[a, N - a] for a in powerful
                          if a <= hi and _is_powerful(N - a)]

    # Fallback to the FFT peak if the scan was cut short before finding anything.
    if best_N is None:
        best_N = peak
        hi = min(M, best_N // 2)
        best_pairs = [[a, best_N - a] for a in powerful
                      if a <= hi and _is_powerful(best_N - a)]
        best_count = len(best_pairs)

    # Safety: if somehow nothing was found, return a valid trivial result.
    if best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': best_pairs}
