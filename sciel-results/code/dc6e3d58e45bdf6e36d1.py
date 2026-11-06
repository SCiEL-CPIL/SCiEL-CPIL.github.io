"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    DEADLINE = time.time() + 145.0  # safety margin before 150s kill
    MAX_K = 200

    # --- Deterministic Miller-Rabin for n < 3.3e24 ---
    _MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for p in _MR_BASES:
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
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # --- Pollard's Rho (Brent variant) ---
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randrange(2, n - 1)
            y = x
            d = 1
            c = random.randrange(1, n - 1)
            m = 128
            q = 1
            g = 1
            while g == 1:
                for _ in range(m):
                    x = (x * x + c) % n
                    q = q * abs(x - y) % n
                y = x
                g = math.gcd(q, n)
                if g == n:
                    break
            if g == n:
                g = 1
                while g == 1:
                    x = (x * x + c) % n
                    g = math.gcd(abs(x - y), n)
                if g == n:
                    continue
            if g != n:
                return g

    def factor(n, result):
        """Recursively factor n, appending prime factors to result."""
        if n == 1:
            return
        if is_prime(n):
            result.append(n)
            return
        d = pollard_rho(n)
        factor(d, result)
        factor(n // d, result)

    # --- Sieve of Eratosthenes for small primes ---
    def sieve_primes(limit):
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
        return [i for i in range(limit + 1) if is_p[i]]

    # Pre-compute primes up to a reasonable bound for trial division
    SMALL_PRIME_LIMIT = 100000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)

    def is_p_smooth(n, p, small_primes):
        """Check if all prime factors of n are <= p using trial division then Pollard-Brent."""
        # Trial divide by small primes up to min(p, SMALL_PRIME_LIMIT)
        for sp in small_primes:
            if sp > p:
                break
            if sp * sp > n:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        if n > p:
            # Need to check if n's factors are all <= p
            # If n is prime and n > p, fail
            if is_prime(n):
                return n <= p
            # Factor n completely
            factors = []
            factor(n, factors)
            return all(f <= p for f in factors)
        else:
            # n <= p, but n might be composite with factors > p? No, if n <= p then all factors <= n <= p
            return True

    def compute_k(p, small_primes, deadline):
        """Compute the maximal k such that p^2+1 ... p^2+k are all p-smooth."""
        k = 0
        base = p * p
        while k < MAX_K:
            if time.time() > deadline:
                break
            n = base + k + 1
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
        return k

    # --- Generate candidate primes ---
    # We want to search primes up to some limit. Use a segmented sieve to generate primes.
    best_k = 0
    best_p = 2

    # First check p=2
    if time.time() < deadline:
        k2 = compute_k(2, small_primes, deadline)
        if k2 > best_k:
            best_k = k2
            best_p = 2

    # Generate primes using segmented sieve up to a limit
    # Let's try primes up to 50 million first, then expand if time allows
    SIEVE_LIMIT = 50_000_000

    def segmented_sieve(low, high, small_primes):
        """Generate primes in [low, high] using segmented sieve."""
        size = high - low + 1
        is_prime_seg = bytearray([1]) * size
        if low == 1:
            is_prime_seg[0] = 0
        for p in small_primes:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                is_prime_seg[j - low] = 0
        return [low + i for i in range(size) if is_prime_seg[i]]

    # Process primes in segments
    SEGMENT_SIZE = 5_000_000
    low = 3
    while low <= SIEVE_LIMIT and time.time() < deadline:
        high = min(low + SEGMENT_SIZE - 1, SIEVE_LIMIT)
        primes_in_seg = segmented_sieve(low, high, small_primes)
        for p in primes_in_seg:
            if time.time() > deadline:
                return {'p': best_p, 'k': best_k}
            k = compute_k(p, small_primes, deadline)
            if k > best_k:
                best_k = k
                best_p = p
        low = high + 1

    # If we have time, extend the search to larger primes
    # But first, let's also try some specific large primes that might be good
    # Actually, let's just extend the segmented sieve further if time permits
    EXTENDED_LIMIT = 200_000_000
    low = SIEVE_LIMIT + 1
    while low <= EXTENDED_LIMIT and time.time() < deadline:
        high = min(low + SEGMENT_SIZE - 1, EXTENDED_LIMIT)
        primes_in_seg = segmented_sieve(low, high, small_primes)
        for p in primes_in_seg:
            if time.time() > deadline:
                return {'p': best_p, 'k': best_k}
            k = compute_k(p, small_primes, deadline)
            if k > best_k:
                best_k = k
                best_p = p
        low = high + 1

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
