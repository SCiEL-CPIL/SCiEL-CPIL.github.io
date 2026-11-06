"""Rewritten program: structural change to a batched, sieve-accelerated search over primes."""
import time
import math
import random

_DEADLINE_FRACTION = 0.85  # stop searching at 85% of the 150 s budget to be safe

# ---------- deterministic Miller–Rabin for n < 3.3e24 ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard–Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
        m = 128
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

def _factorize(n, out):
    """Recursively factor n, appending prime factors to out."""
    if n == 1:
        return
    if _is_prime_mr(n):
        out.append(n)
        return
    d = _pollard_brent(n)
    _factorize(d, out)
    _factorize(n // d, out)

# ---------- small-prime sieve (up to 10^6) for fast trial division ----------
_SIEVE_LIMIT = 1_000_000

def _sieve_primes(limit):
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit + 1:step] = bytearray(len(range(start, limit + 1, step)))
    return [i for i, v in enumerate(is_prime) if v]

_SMALL_PRIMES = _sieve_primes(_SIEVE_LIMIT)

# ---------- batched window evaluation ----------
def _evaluate_window(p, max_k):
    """
    Return the largest k in [0, max_k] such that every integer
    p^2+1 .. p^2+k has all prime factors <= p.
    Uses a batched sieve over the window with all primes <= p.
    """
    k = max_k
    if k < 1:
        return 0
    start = p * p + 1
    end = start + k  # exclusive
    # residues array: 0 means "not yet cleared"
    res = [0] * k

    # Determine which primes to use for sieving: all primes <= p
    # We need primes up to p.  For p up to ~1e6 we have _SMALL_PRIMES.
    # For larger p we need to generate more primes or rely on Pollard for the remainder.
    # Strategy: sieve with primes up to min(p, _SIEVE_LIMIT) first, then check the
    # remaining cofactor with MR + Pollard.

    # Sieve with small primes
    for q in _SMALL_PRIMES:
        if q > p:
            break
        # first multiple of q in [start, end)
        first = ((start + q - 1) // q) * q
        if first >= end:
            continue
        for m in range(first, end, q):
            idx = m - start
            while res[idx] == 0 or res[idx] % q == 0:
                if res[idx] == 0:
                    res[idx] = q
                elif res[idx] % q == 0:
                    res[idx] //= q
                else:
                    break
    # After sieving with small primes, for each position i, res[i] is either:
    #   0  -> the number was not touched (shouldn't happen for composite)
    #   1  -> fully factored into small primes (all <= p)
    #   >1 -> remaining cofactor; need to check if it's <= p or prime <= p

    # Now check each position
    max_valid = 0
    for i in range(k):
        cof = res[i]
        if cof == 0:
            # number p^2+1+i was not divisible by any small prime; it's either 1 or prime
            # Actually if it's 1 that means it was fully divided; but we init to 0.
            # Let's handle: the original number n = start + i
            n = start + i
            if n == 1:
                continue
            if _is_prime_mr(n):
                if n <= p:
                    max_valid = i + 1
                    continue
                else:
                    break
            else:
                # composite with no small prime factor -> has a prime factor > _SIEVE_LIMIT
                # must be > p if p <= _SIEVE_LIMIT, but if p > _SIEVE_LIMIT we need to check
                # For safety, factor it
                factors = []
                _factorize(n, factors)
                if all(f <= p for f in factors):
                    max_valid = i + 1
                else:
                    break
        elif cof == 1:
            max_valid = i + 1
        else:
            # cof > 1: check if cof <= p (then OK) or if cof is prime and <= p
            if cof <= p:
                max_valid = i + 1
            else:
                # cof is a composite or prime > p; need to verify
                # If cof is prime, it must be <= p to be valid
                if _is_prime_mr(cof):
                    if cof <= p:
                        max_valid = i + 1
                    else:
                        break
                else:
                    # cof is composite; factor it to check all prime factors
                    factors = []
                    _factorize(cof, factors)
                    if all(f <= p for f in factors):
                        max_valid = i + 1
                    else:
                        break
    return max_valid

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 * _DEADLINE_FRACTION

    best_k = 1
    best_p = 2

    # We'll search primes in increasing order, but skip to larger ranges as time allows.
    # For small p, the window is short; for larger p, windows can be longer.
    # Start with primes up to 1e6, then extend if time permits.

    # Generate primes up to a growing limit
    search_limit = 1_000_000
    # Use the precomputed _SMALL_PRIMES for primes up to 1e6
    primes_to_try = _SMALL_PRIMES[:]

    # If we have time, extend the prime list
    while time.time() < deadline:
        # Try the next chunk of primes
        if not primes_to_try:
            break

        # Process primes in batches
        batch_size = 5000
        for i in range(0, len(primes_to_try), batch_size):
            if time.time() >= deadline:
                break
            batch = primes_to_try[i:i+batch_size]
            for p in batch:
                if time.time() >= deadline:
                    break
                if p <= 2:
                    continue
                # Quick estimate: try a window
                # For small p, k is small; for larger p, k can be bigger.
                # Start with a modest max_k and grow if needed.
                max_k = min(200, max(10, p // 10))
                k = _evaluate_window(p, max_k)
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
