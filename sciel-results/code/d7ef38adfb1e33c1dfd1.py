import time
import math
import random
from functools import lru_cache

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
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
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
            if g == n:
                continue
        return g

# ---------- Factorization with cache ----------
_factor_cache = {}

def _factor(n):
    if n in _factor_cache:
        return _factor_cache[n]
    if n == 1:
        _factor_cache[n] = ()
        return ()
    if _is_prime_mr(n):
        _factor_cache[n] = (n,)
        return (n,)
    d = _pollard_brent(n)
    result = _factor(d) + _factor(n // d)
    _factor_cache[n] = result
    return result

def _all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    if _is_prime_mr(n):
        return n <= p
    factors = _factor(n)
    return all(f <= p for f in factors)

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    best_k = 0
    best_p = 2

    # Precompute primes up to 2M for segmented sieve and pre-filter
    SMALL_LIMIT = 2_000_000
    small_primes = sieve_primes(SMALL_LIMIT)
    
    # For pre-filter: primes up to 1M (q > p candidates)
    PREFILTER_LIMIT = 1_000_000
    prefilter_primes = [q for q in small_primes if q <= PREFILTER_LIMIT]

    def check_run_segmented(p, max_k=200):
        """
        Check p^2+1 .. p^2+max_k are all p-smooth using segmented sieve.
        
        Step 1: Sieve out all prime factors <= p from the interval.
        Step 2: For remaining values > p, factor and check all factors <= p.
        """
        p_sq = p * p
        # Segment: [p_sq+1, p_sq+max_k]
        segment_size = max_k
        # rem[i] holds the remaining part of (p_sq + i) after dividing by primes <= p
        rem = [p_sq + i for i in range(1, segment_size + 1)]
        
        # Sieve: divide out all primes q <= p
        for q in small_primes:
            if q > p:
                break
            # Find first index i in [1, max_k] such that q | (p_sq + i)
            # i.e., i ≡ -p_sq (mod q), i.e., i ≡ -(p^2 mod q) (mod q)
            p2_mod_q = (p * p) % q
            target = (-p2_mod_q) % q  # i mod q should equal this
            if target == 0:
                target = q
            # First i >= 1 with i ≡ target (mod q)
            if target >= 1:
                first = target
            else:
                first = target + q  # shouldn't happen since target in [0,q]
            # Actually target is in [0, q-1], and we want i in [1, segment_size]
            # i = target if target >= 1, else i = q
            if target == 0:
                first = q
            else:
                first = target
            if first > segment_size:
                continue
            # Divide out all powers of q from rem[first], rem[first+q], ...
            for idx in range(first - 1, segment_size, q):
                while rem[idx] % q == 0:
                    rem[idx] //= q
        
        # Now check each position
        k = 0
        for i in range(1, segment_size + 1):
            r = rem[i - 1]
            if r == 1:
                k += 1
                continue
            if r <= p:
                # r is either 1 (handled) or a prime <= p, or composite with all factors <= p
                # If r <= p and r > 1, r must be prime (since we divided out all q <= p)
                # Wait - r could be composite if it's a product of primes all > p? No, r <= p means
                # all its factors are <= p. But we already divided out all primes <= p, so r must be 1
                # or r is prime > p... contradiction with r <= p.
                # Actually: after sieving, r has no prime factors <= p. If r <= p and r > 1,
                # then r is prime and r > p... but r <= p. Contradiction. So r must be 1.
                # Unless r is a prime power where the base > p but the power brings it below p?
                # No: if q > p divides r, then r >= q > p. So r <= p implies r = 1.
                k += 1
                continue
            # r > p: need to check if all prime factors of r are <= p
            # But r has no prime factors <= p (we sieved them out)!
            # So r must be prime (and > p) or a product of primes all > p.
            # In either case, r has a prime factor > p.
            # Therefore r > p means NOT p-smooth.
            break
        
        return k

    def check_run_full(p, max_k=200):
        """
        Full check: for each n = p^2+i, factor n and verify all factors <= p.
        Uses the segmented sieve to remove small factors, then factors the remainder.
        """
        p_sq = p * p
        segment_size = max_k
        rem = [p_sq + i for i in range(1, segment_size + 1)]
        
        # Sieve out primes q <= p
        for q in small_primes:
            if q > p:
                break
            p2_mod_q = (p * p) % q
            target = (-p2_mod_q) % q
            if target == 0:
                first = q
            else:
                first = target
            if first > segment_size:
                continue
            for idx in range(first - 1, segment_size, q):
                while rem[idx] % q == 0:
                    rem[idx] //= q
        
        k = 0
        for i in range(1, segment_size + 1):
            r = rem[i - 1]
            if r == 1:
                k += 1
                continue
            # r > 1: r has no prime factors <= p
            # Check if r is prime
            if _is_prime_mr(r):
                # r is prime and r > p (since no factors <= p), so NOT p-smooth
                break
            # r is composite with no factors <= p, so has a factor > p
            # NOT p-smooth
            break
        
        return k

    def prefilter(p):
        """
        Quick rejection: check if any prime q in (p, PREFILTER_LIMIT] divides p^2+i for some i in [1,200].
        If so, that n has a factor > p, so the run is broken at that point.
        Returns the minimum i in [1,200] where p^2+i is divisible by some q in (p, PREFILTER_LIMIT],
        or 201 if no such collision found.
        """
        min_bad = 201
        for q in prefilter_primes:
            if q <= p:
                continue
            # Check if p^2 + i ≡ 0 (mod q) for some i in [1, 200]
            # i ≡ -p^2 (mod q)
            p2_mod_q = (p * p) % q
            target = (-p2_mod_q) % q
            if target == 0:
                i = q
            else:
                i = target
            if 1 <= i <= 200:
                if i < min_bad:
                    min_bad = i
        return min_bad

    def verify_run(p, k):
        """Full verification that all of p^2+1..p^2+k are p-smooth."""
        p_sq = p * p
        for i in range(1, k + 1):
            n = p_sq + i
            if not _all_factors_at_most_p(n, p):
                return False
        return True

    # Phase 1: Search small primes up to 10^6 with segmented sieve
    search_limit = 1_000_000
    primes_to_test = [p for p in small_primes if p <= search_limit]

    for p in primes_to_test:
        if time.time() > deadline:
            break
        # Pre-filter: if any q in (p, 1M] hits, the run is at most min_bad-1
        pf = prefilter(p)
        max_possible = min(200, pf - 1)
        if max_possible <= best_k:
            continue
        k = check_run_full(p, max_possible)
        if k > best_k:
            if verify_run(p, k):
                best_k = k
                best_p = p

    # Phase 2: Search larger primes using segmented sieve
    if time.time() < deadline - 15:
        # Generate primes in chunks using Miller-Rabin
        chunk_size = 200_000
        for start in range(1_000_000, 30_000_000, chunk_size):
            if time.time() > deadline:
                break
            end = min(start + chunk_size, 30_000_000)
            for candidate in range(start, end, 2):
                if candidate % 2 == 0:
                    continue
                if not _is_prime_mr(candidate):
                    continue
                # Pre-filter
                pf = prefilter(candidate)
                max_possible = min(200, pf - 1)
                if max_possible <= best_k:
                    continue
                k = check_run_full(candidate, max_possible)
                if k > best_k:
                    if verify_run(candidate, k):
                        best_k = k
                        best_p = candidate

    # Phase 3: Random sampling of larger primes
    if time.time() < deadline - 5:
        for _ in range(300):
            if time.time() > deadline:
                break
            p = random.randint(10**6, 5*10**7)
            if p % 2 == 0:
                p += 1
            while not _is_prime_mr(p):
                p += 2
            pf = prefilter(p)
            max_possible = min(200, pf - 1)
            if max_possible <= best_k:
                continue
            k = check_run_full(p, max_possible)
            if k > best_k:
                if verify_run(p, k):
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
