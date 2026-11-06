import math
import time

def construct():
    start = time.time()
    deadline = start + 55.0

    N = 123457
    half = N // 2  # 61728

    # ---------- Brent's integer factorization (largest prime factor) ----------
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        if n % 3 == 0:
            return n == 3
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def _brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        y, c, m = 2, 1, 128
        g, r, q = 1, 1, 1
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
                g = _gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = _gcd(abs(x - ys), n)
                if g > 1:
                    break
        return g

    def _factor_rec(n, factors):
        if n == 1:
            return
        d = _brent(n)
        _factor_rec(d, factors)
        _factor_rec(n // d, factors)

    def largest_prime_factor(n):
        if n <= 1:
            return 1
        factors = []
        _factor_rec(n, factors)
        return max(factors)

    # ---------- Trial-division LPF with early exit ----------
    # Primes up to 400 (400^2 > 123457)
    _primes = []
    for _p in range(2, 401):
        if all(_p % d for d in range(2, int(_p ** 0.5) + 1)):
            _primes.append(_p)

    def lpf_trial(n, limit):
        """Largest prime factor of n; returns > limit early if found."""
        if n <= 1:
            return 1
        result = 1
        for p in _primes:
            if p > limit:
                return p  # p divides nothing yet but p > limit means we'd exceed
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result = p
                if result > limit:
                    return result
        if n > 1:
            result = n
        return result

    # ---------- Phase 1: trial-division scan for a quick good baseline ----------
    best_val = float('inf')
    best_a = 1

    for a in range(1, half + 1):
        b = N - a
        # Early exit: if either factor already exceeds best, skip
        pa = lpf_trial(a, best_val)
        if pa > best_val:
            continue
        pb = lpf_trial(b, best_val)
        if pb > best_val:
            continue
        cur = pa if pa > pb else pb
        if cur < best_val:
            best_val = cur
            best_a = a

        if time.time() > deadline - 10:
            break

    # ---------- Phase 2: precise Brent-based verification of improving candidates ----------
    # Only candidates with trial-LPF <= best_val can possibly improve; verify with Brent.
    # We scan all a again but only do expensive Brent when trial suggests improvement.
    improved = True
    while improved:
        improved = False
        for a in range(1, half + 1):
            b = N - a
            pa = lpf_trial(a, best_val)
            if pa > best_val:
                continue
            pb = lpf_trial(b, best_val)
            if pb > best_val:
                continue
            # Trial says both <= best_val; verify precisely with Brent
            pa_b = largest_prime_factor(a)
            if pa_b > best_val:
                continue
            pb_b = largest_prime_factor(b)
            if pb_b > best_val:
                continue
            cur = pa_b if pa_b > pb_b else pb_b
            if cur < best_val:
                best_val = cur
                best_a = a
                improved = True

            if time.time() > deadline - 2:
                return [best_a, N - best_a]

    return [best_a, N - best_a]
