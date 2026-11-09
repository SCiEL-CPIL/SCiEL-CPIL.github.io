"""Rewritten program: minimize distinct 24-smooth parts over 24 consecutive integers."""
import time

# EVOLVE-BLOCK-START
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_MAXN = 1000000
_WINDOW = 24
_MINN = 1000


def _precompute_smooth_parts():
    """f[m] = largest divisor of m using only primes < 24, for m up to _MAXN+_WINDOW."""
    limit = _MAXN + _WINDOW
    f = list(range(limit + 1))
    for p in _PRIMES:
        # For each multiple of p, divide out all powers of p
        for m in range(p, limit + 1, p):
            x = m
            while x % p == 0:
                x //= p
            f[m] = x
    return f


def _score_window(f, n):
    """Count distinct f-values over n+1 .. n+24."""
    vals = f[n + 1: n + 1 + _WINDOW]
    return len(set(vals))


def construct():
    t0 = time.time()
    deadline = 60.0
    f = _precompute_smooth_parts()

    # Initial full scan
    best_n = _MINN
    best_score = _score_window(f, _MINN)
    for n in range(_MINN + 1, _MAXN + 1):
        s = _score_window(f, n)
        if s < best_score:
            best_score = s
            best_n = n
        if time.time() - t0 > deadline - 5.0:
            break

    # If best is already very low, try to confirm / refine with a focused rescan
    # (cheap) and return early once we hit a strong lower bound.
    if best_score <= 3 and time.time() - t0 < deadline - 10.0:
        # Rescan to ensure we didn't miss a better one due to time cutoff
        for n in range(_MINN, _MAXN + 1):
            s = _score_window(f, n)
            if s < best_score:
                best_score = s
                best_n = n
            if time.time() - t0 > deadline - 5.0:
                break

    return best_n
# EVOLVE-BLOCK-END
