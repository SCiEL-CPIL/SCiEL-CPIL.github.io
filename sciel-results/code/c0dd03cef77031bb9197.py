"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """Find a prime p <= 1e9 maximizing k such that p^2+1..p^2+k are all p-smooth."""
    deadline = time.monotonic() + 110  # safety margin before 120s limit
    best_k = 1
    best_p = 3

    # --- Sieve primes up to 1e6 for trial division ---
    B = 10**6
    sieve = bytearray(b'\x01') * (B + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(B)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, B + 1) if sieve[i]]
    small_primes_set = set(small_primes)

    # --- Deterministic Miller-Rabin for n < 3.3e24 ---
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # --- Pollard's Rho for finding a factor of n ---
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = 2
            y = 2
            d = 1
            c = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            c += 1

    # --- Factor n completely (returns list of prime factors) ---
    def factor(n, factors=None):
        if factors is None:
            factors = []
        if n <= 1:
            return factors
        # Trial division by small primes
        for p in small_primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        d = pollard_rho(n)
        factor(d, factors)
        factor(n // d, factors)
        return factors

    # --- Check if n is p-smooth (all prime factors <= p) ---
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        # Trial division by primes <= min(p, B)
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        if n == 1:
            return True
        if n <= p:
            return True  # n is prime and <= p
        # n > p, check if n is prime
        if is_prime_mr(n):
            return False  # n is prime > p
        # n is composite and > p; factor it
        facs = factor(n)
        return all(f <= p for f in facs)

    # --- Check run length for a given prime p ---
    def check_run(p):
        k = 0
        base = p * p
        while k < 200:
            if time.monotonic() > deadline:
                return k
            n = base + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    # --- Generate candidate primes and evaluate ---
    # Strategy: iterate through candidate p values, prioritizing larger p
    # Start from a high range and work down, using a wheel to skip non-candidates.

    # Wheel: skip multiples of 2, 3, 5
    wheel = [6, 4, 2, 4, 2, 4, 6, 2]  # increments to skip composites
    wheel_idx = 0

    def next_candidate(current):
        """Return next candidate after current that is not divisible by 2,3,5."""
        c = current + 1
        while c % 2 == 0 or c % 3 == 0 or c % 5 == 0:
            c += 1
        return c

    # Search strategy: try several ranges of p, from larger to smaller
    # For each range, check candidates and track best (k, p)

    search_ranges = [
        (999_999_999, 900_000_000),
        (900_000_000, 800_000_000),
        (800_000_000, 700_000_000),
        (700_000_000, 600_000_000),
        (600_000_000, 500_000_000),
        (500_000_000, 400_000_000),
        (400_000_000, 300_000_000),
        (300_000_000, 200_000_000),
        (200_000_000, 100_000_000),
        (100_000_000, 50_000_000),
        (50_000_000, 10_000_000),
        (10_000_000, 1_000_000),
    ]

    for hi, lo in search_ranges:
        if time.monotonic() > deadline:
            break

        # Find the first candidate at or below hi
        c = hi
        if c % 2 == 0 or c % 3 == 0 or c % 5 == 0:
            c -= 1
            while c % 2 == 0 or c % 3 == 0 or c % 5 == 0:
                c -= 1

        checked = 0
        while c >= lo and time.monotonic() < deadline:
            # Check if c is prime
            if is_prime_mr(c):
                k = check_run(c)
                if k > best_k:
                    best_k = k
                    best_p = c
                    # If we found a very good k, we can try to refine further
                    if k >= 200:
                        return {'p': c, 'k': 200}
            c -= 2
            # Skip to next candidate not divisible by 3 or 5
            while c >= lo and (c % 3 == 0 or c % 5 == 0):
                c -= 2
            checked += 1

    # Fallback: ensure we have at least p=3, k=1
    if best_k < 1:
        best_k = 1
        best_p = 3

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
