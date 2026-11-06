import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # ── Precomputed small primes ──────────────────────────────────────────────
    _SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                     41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                     101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                     151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
                     211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
                     271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337,
                     347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401,
                     409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                     467, 479, 487, 491, 499]
    _SMALL_PRIME_SET = set(_SMALL_PRIMES)

    # ── Deterministic Miller-Rabin for n < 2^64 ───────────────────────────────
    _MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for p in _SMALL_PRIMES:
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

    # ── Pollard's Rho with Brent's cycle detection ────────────────────────────
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                if g == n:
                    continue  # retry with new parameters
            return g

    # ── Factorization ──────────────────────────────────────────────────────────
    def factorize(n, factors=None):
        """Recursively factor n, appending prime factors to *factors*."""
        if factors is None:
            factors = []
        if n == 1:
            return factors
        for p in _SMALL_PRIMES:
            if n % p == 0:
                factors.append(p)
                factorize(n // p, factors)
                return factors
        if is_prime(n):
            factors.append(n)
            return factors
        d = pollard_rho(n)
        factorize(d, factors)
        factorize(n // d, factors)
        return factors

    def largest_prime_factor(n):
        if n == 1:
            return 1
        factors = factorize(n)
        return max(factors)

    # ── Segmented sieve for prime generation in arbitrary ranges ──────────────
    def segmented_sieve(low, high):
        """Return list of primes in [low, high]."""
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        # Base sieve up to limit
        base = bytearray(b'\x01') * (limit + 1)
        base[0] = base[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if base[i]:
                base[i*i::i] = bytearray(len(base[i*i::i]))
        small_primes = [i for i in range(2, limit + 1) if base[i]]

        size = high - low + 1
        # Ensure low is at least 2
        if low < 2:
            low = 2
            size = high - low + 1
        seg = bytearray(b'\x01') * size
        for p in small_primes:
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start - low, size, p):
                seg[j] = 0
        primes = []
        for i in range(size):
            if seg[i]:
                primes.append(low + i)
        return primes

    # ── Check how many consecutive values p^2+1..p^2+k are p-smooth ───────────
    def check_smooth_run(p, max_k=200):
        """Return the length of the p-smooth run starting at p^2+1."""
        k = 0
        p2 = p * p
        for i in range(1, max_k + 1):
            n_val = p2 + i
            # If n_val <= p, it's trivially p-smooth (all factors <= n_val <= p)
            if n_val <= p:
                k += 1
                continue
            # Quick prime check: if n_val is prime and > p, fail immediately
            if is_prime(n_val):
                break
            # Find largest prime factor
            lf = largest_prime_factor(n_val)
            if lf > p:
                break
            k += 1
        return k

    # ── Main search ────────────────────────────────────────────────────────────
    best_k = 0
    best_p = 2

    # Quick check small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if time.time() > deadline:
            break
        k = check_smooth_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Systematic search using segmented sieve
    # Start from 40 (after small primes) up to as high as time permits
    # Use segments of size ~10^6 to keep memory bounded
    SEG_SIZE = 10**6
    current_low = 40

    while current_low < 10**9:
        if time.time() > deadline - 5:
            break

        current_high = min(current_low + SEG_SIZE - 1, 10**9)
        primes_in_seg = segmented_sieve(current_low, current_high)

        for p in primes_in_seg:
            if time.time() > deadline - 2:
                break

            # Quick rejection: p^2+1 must not be prime (or if prime, <= p which is impossible for p>=2)
            n1 = p * p + 1
            if is_prime(n1):
                continue

            k = check_smooth_run(p)
            if k > best_k:
                best_k = k
                best_p = p

        current_low = current_high + 1

    # If time remains, try some random large primes beyond what we've covered
    if time.time() < deadline - 10:
        random.seed(42)
        tried = set()
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            p_candidate = random.randint(10**6, 10**9)
            if p_candidate % 2 == 0:
                p_candidate += 1
            if p_candidate in tried:
                continue
            tried.add(p_candidate)
            if is_prime(p_candidate):
                n1 = p_candidate * p_candidate + 1
                if is_prime(n1):
                    continue
                k = check_smooth_run(p_candidate)
                if k > best_k:
                    best_k = k
                    best_p = p_candidate

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
