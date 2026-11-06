import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118.0

    # Sieve of Eratosthenes using bytearray for speed
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Precompute primes up to 1,000,000 for fast trial division
    SIEVE_LIMIT = 1000000
    primes = sieve_primes(SIEVE_LIMIT)
    num_primes = len(primes)

    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # Pollard Rho with multiple starting points for robustness
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        c = 1
        while True:
            x, y, d = 2, 2, 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            c += 1
            if c > 100:
                # Try different starting points
                x, y, d = 3, 3, 1
                c = 1
                while d == 1:
                    x = (x * x + c) % n
                    y = (y * y + c) % n
                    y = (y * y + c) % n
                    d = math.gcd(abs(x - y), n)
                if d != n:
                    return d

    # Check if n is p-smooth using trial division first, then Pollard-Rho
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        m = n
        # Trial division by small primes
        for q in primes:
            if q * q > m:
                break
            if q > p:
                break
            if m % q == 0:
                while m % q == 0:
                    m //= q
                if m == 1:
                    return True
        if m == 1:
            return True
        if m <= p:
            return True
        # m > p and m is not fully factored by small primes
        # Check if m is prime
        if is_prime_mr(m):
            return False  # m is prime and > p
        # Use Pollard-Rho to factor m
        d = pollard_rho(m)
        if d == m:
            # Failed to factor, assume it has a large prime factor
            return is_prime_mr(m) and m <= p
        return is_p_smooth(d, p) and is_p_smooth(m // d, p)

    # Check run length for a given prime p
    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 1:
                break
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    best_k = 1
    best_p = 2

    # Phase 1: Check small primes quickly (p <= 100,000)
    # These are fast because trial division is cheap
    for p in primes:
        if p > 100000:
            break
        if time.time() > DEADLINE - 10:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Check medium primes (100,000 < p <= 1,000,000)
    # Still use trial division, but fewer candidates
    for p in primes:
        if p > 1000000:
            break
        if p <= 100000:
            continue
        if time.time() > DEADLINE - 5:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 3: For larger primes, use a smarter approach
    # Only check primes that pass a quick pre-filter
    # Pre-filter: check if p^2 + 1 is p-smooth (necessary condition)
    # If not, skip entirely
    
    # Generate more primes up to 10^7 for phase 3
    SIEVE_LIMIT2 = 10000000
    if time.time() < DEADLINE - 20:
        primes2 = sieve_primes(SIEVE_LIMIT2)
        for p in primes2:
            if p > 10000000:
                break
            if p <= 1000000:
                continue
            if time.time() > DEADLINE - 3:
                break
            # Quick pre-filter: check p^2 + 1
            n1 = p * p + 1
            if not is_p_smooth(n1, p):
                continue
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
