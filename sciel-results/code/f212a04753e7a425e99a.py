import time
import math
import random

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

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
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
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
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

# ---------- Sieve ----------
def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute small primes up to 10^6 for fast trial division
    small_primes = _sieve(1000000)

    best_k = 0
    best_p = 3

    # ---- Phase 1: Exhaustive search over small primes (p <= 10^6) ----
    # For small p, trial division is fast; check all 200 values.
    for p in small_primes:
        if time.time() > deadline:
            break
        if p > 10**6:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            t = p_sq + i
            for q in small_primes:
                if q > p:
                    break
                if q * q > t:
                    break
                while t % q == 0:
                    t //= q
            if t == 1:
                k = i
                continue
            if t <= p:
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': best_p, 'k': best_k}

    # ---- Phase 2: Batched pre-filter for larger primes ----
    # Key idea: for a prime p, p^2+i must be p-smooth for i=1..k.
    # If there exists a prime q with p/2 < q <= p such that q | (p^2+i) for some i in [1,200],
    # then p^2+i has a factor q > p/2. Since p^2+i < p^2+200 and q > p/2,
    # the cofactor (p^2+i)/q < 2p + 400/q < 2p + 800.
    # If the cofactor is also > p (which happens when p^2+i > p*q, i.e., i > p*q - p^2 = p(q-p)),
    # then p^2+i is NOT p-smooth, so k < i.
    #
    # More precisely: if q | (p^2+i) and q > p/2, then for p^2+i to be p-smooth,
    # we need (p^2+i)/q to be p-smooth AND <= p^2 (always true).
    # But if (p^2+i)/q > p, it could still be p-smooth (product of smaller primes).
    # However, if (p^2+i)/q is prime and > p, then p^2+i is NOT p-smooth.
    #
    # Simpler sufficient condition for rejection: if q | (p^2+i) for some i in [1,200]
    # and q > p/2, then the remaining factor r = (p^2+i)/q satisfies:
    #   r = (p^2+i)/q. Since q > p/2, r < 2p + 400/q < 2p + 800.
    #   If r is prime and r > p, then p^2+i is not p-smooth → k < i.
    #   Even if r is composite, if r has a prime factor > p, same conclusion.
    #
    # For the pre-filter, we use a necessary condition: if for EVERY i in [1,200],
    # there exists some prime q in (p/2, p] with q | (p^2+i), AND for at least one such i,
    # the cofactor is provably > p and prime, then k < i.
    #
    # A cheaper necessary condition: if there's a prime q in (p/2, p] that divides p^2+i
    # for some i in [1,200], and (p^2+i)//q > p, then p^2+i is likely not p-smooth.
    # This isn't a proof, but it's a very strong heuristic filter.
    #
    # Actually, let's use a cleaner approach:
    # For p to have k >= K, we need p^2+i to be p-smooth for all i in [1,K].
    # A necessary condition: for all i in [1,K], there is NO prime q in (p/2, p] with q | (p^2+i).
    #   (Because if q | (p^2+i) and q > p/2, then p^2+i = q * r where r < 2p+400.
    #    If r > p, then either r is prime > p (not p-smooth) or r has a factor > p.
    #    If r <= p, then p^2+i = q*r with both factors <= p, so it IS p-smooth.
    #    So the condition is: for all i in [1,K], either no prime q in (p/2,p] divides p^2+i,
    #    or (p^2+i)/q <= p.)
    #
    # Even simpler: for all i in [1,K], the smallest prime factor of (p^2+i) that is > p/2
    # must satisfy (p^2+i)/spf <= p. Equivalently, spf >= (p^2+i)/p = p + i/p > p.
    # So if there's a prime factor q with p/2 < q < (p^2+i)/p = p + i/p, then p^2+i is not p-smooth.
    # Since i/p < 1 for i < p, this means: if there's a prime factor q with p/2 < q < p,
    # then p^2+i is not p-smooth (for i < p, which is always true since i <= 200 << p).
    #
    # WAIT: that's wrong. If q | (p^2+i) and p/2 < q < p, then r = (p^2+i)/q.
    # r could be <= p (if q >= (p^2+i)/p ≈ p), or r > p.
    # If r > p and r is prime, not p-smooth. If r > p and composite, might still be p-smooth.
    #
    # Let me just use the direct check: for each i, after trial division by primes <= p/2,
    # the remainder must be <= p (and prime) or 1. If remainder > p, factor it and check.
    # The speedup comes from: trial dividing only by primes <= p/2 (half the primes),
    # and using the fact that if remainder > p, it's likely prime (and > p → fail).

    # For the batched approach, let's precompute for a range of candidate p values:
    # For each small prime q (up to ~10^6), compute p^2 mod q for many p values.
    # If p^2 ≡ -i (mod q) for some i in [1,200], then q | (p^2+i).
    # If q > p/2, this is a "bad" hit.

    # Strategy: for each candidate p (found via random search in [10^6, 10^9]),
    # do a fast pre-check using primes in (p/2, p] ∩ [2, 10^6] (only possible if p <= 2*10^6).
    # For p > 2*10^6, all primes in (p/2, p] are > 10^6, so we can't pre-check them with our sieve.
    # In that case, fall back to direct trial division + Pollard-Brent.

    # Actually, the most effective approach for large p:
    # 1. Trial divide p^2+i by primes up to min(p, 10^6).
    # 2. If remainder is 1 or <= p, it's p-smooth (for that i).
    # 3. If remainder > p, check if it's prime (MR). If prime > p, fail.
    # 4. If composite, Pollard-Brent factor it and check all factors <= p.
    #
    # The key optimization: for large p (> 10^6), step 1 only needs primes up to 10^6,
    # and the remainder is at most (p^2+200)/(product of small prime factors).
    # For random p, most p^2+i will have a large prime factor > p, so we fail quickly.
    # The challenge is finding the rare p where ALL 200 values are p-smooth.

    # Let's try a different tactic: search primes in increasing order from 10^6 upward,
    # but use a very fast rejection test. For each p, check i=1 first (cheapest),
    # then i=2, etc. Most primes will fail at small i.

    # Also: use the property that p^2+1, p^2+2, ... are consecutive integers.
    # Among any 200 consecutive integers, there are many with small prime factors.
    # The constraint is that NO prime factor exceeds p. For large p, this is easier
    # because the "forbidden" range (p, ∞) is a smaller fraction of the factor space.

    # Let's do a structured search: try primes in blocks, with early termination.

    # Phase 2a: primes from 10^6 to 10^7 (dense search)
    # Phase 2b: primes from 10^7 to 10^8 (sparser)
    # Phase 2c: primes from 10^8 to 10^9 (sparse random)

    def _check_prime(p, small_primes, deadline):
        """Check how many consecutive values p^2+1, p^2+2, ... are p-smooth."""
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            t = p_sq + i
            # Trial divide by small primes up to min(p, 10^6)
            for q in small_primes:
                if q > p:
                    break
                if q * q > t:
                    break
                while t % q == 0:
                    t //= q
            if t == 1:
                k = i
                continue
            # t is the remaining factor after trial division
            if t <= p:
                # t is a prime (or 1) <= p, so p-smooth
                k = i
                continue
            # t > p: need to check if t is p-smooth
            if _is_prime_mr(t):
                # t is prime > p → not p-smooth
                break
            # t is composite > p: factor it
            factors = []
            _factor(t, factors)
            if all(f <= p for f in factors):
                k = i
            else:
                break
        return k

    # Phase 2a: Dense search from 10^6 to 10^7
    # Generate primes in this range using a segmented sieve
    def _segmented_sieve(lo, hi, small_primes):
        """Generate primes in [lo, hi] using segmented sieve."""
        size = hi - lo + 1
        is_prime = bytearray(b'\x01') * size
        for p in small_primes:
            if p * p > hi:
                break
            start = ((lo + p - 1) // p) * p
            if start < p * p:
                start = p * p
            for j in range(start - lo, size, p):
                is_prime[j] = 0
        return [lo + i for i in range(size) if is_prime[i] and lo + i >= 2]

    # Search primes from 10^6 to 10^7
    if best_k < 200:
        lo = 10**6
        hi = 10**7
        seg_primes = _segmented_sieve(lo, hi, small_primes)
        for p in seg_primes:
            if time.time() > deadline:
                break
            k = _check_prime(p, small_primes, deadline)
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    return {'p': best_p, 'k': best_k}

    # Phase 2b: Search primes from 10^7 to 10^8 (sampled)
    if best_k < 200 and time.time() < deadline:
        lo = 10**7
        hi = 10**8
        # Sample primes in this range
        num_samples = 5000
        for _ in range(num_samples):
            if time.time() > deadline:
                break
            p = random.randint(lo, hi)
            if p % 2 == 0:
                p += 1
            # Find next prime
            attempts = 0
            while not _is_prime_mr(p) and attempts < 200:
                p += 2
                attempts += 1
            if p > hi or not _is_prime_mr(p):
                continue
            k = _check_prime(p, small_primes, deadline)
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    return {'p': best_p, 'k': best_k}

    # Phase 2c: Search primes from 10^8 to 10^9 (sparse random)
    if best_k < 200 and time.time() < deadline:
        lo = 10**8
        hi = 10**9
        num_samples = 3000
        for _ in range(num_samples):
            if time.time() > deadline:
                break
            p = random.randint(lo, hi)
            if p % 2 == 0:
                p += 1
            attempts = 0
            while not _is_prime_mr(p) and attempts < 200:
                p += 2
                attempts += 1
            if p > hi or not _is_prime_mr(p):
                continue
            k = _check_prime(p, small_primes, deadline)
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    return {'p': best_p, 'k': best_k}

    return {'p': best_p, 'k': best_k}
