"""Search for long runs of consecutive integers with distinct totients up to 10^12."""
import time
import math
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def now():
        return time.time()

    # --- Fast phi using precomputed small primes and Miller-Rabin + Pollard's rho ---

    # Precompute primes up to 10^6 for trial division
    def sieve(n):
        is_prime = bytearray([1]) * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = 0
        return [i for i in range(n+1) if is_prime[i]]

    SMALL_PRIMES = sieve(1000000)
    SMALL_PRIME_SET = set(SMALL_PRIMES)

    # Miller-Rabin for primality test (deterministic for n < 2^64)
    def is_prime_mr(n):
        if n < 2:
            return False
        if n in SMALL_PRIME_SET:
            return True
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        # Write n-1 as 2^r * d
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Test with deterministic witnesses for 64-bit
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # Pollard's rho factorization
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            d = 1
            c = random.randint(1, n - 1)
            f = lambda x: (x * x + c) % n
            while d == 1:
                x = f(x)
                y = f(f(y))
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        """Return list of prime factors (with multiplicity) of n."""
        if n <= 1:
            return []
        factors = []
        # Trial division by small primes
        for p in SMALL_PRIMES:
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
        # Use Pollard's rho for remaining composite
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                factors.append(m)
                continue
            d = pollard_rho(m)
            stack.append(d)
            stack.append(m // d)
        return factors

    def phi_fast(x):
        """Compute Euler's totient of x using exact factorization."""
        if x <= 1:
            return 1 if x == 1 else 0
        result = x
        n = x
        for p in SMALL_PRIMES:
            if p * p > n:
                break
            if n % p == 0:
                result -= result // p
                while n % p == 0:
                    n //= p
        if n > 1:
            # n is prime (or 1)
            if n != 1:
                result -= result // n
        return result

    # --- Search strategy ---
    # The seed found the best run in [1, 20000]. We need to search larger ranges.
    # Key insight: long runs of distinct totients are rare. We should search
    # strategic locations: near multiples of large primes, in smooth regions, etc.
    # 
    # Strategy:
    # 1. First, quickly scan small ranges to get a baseline.
    # 2. Then, use random sampling and targeted search in larger ranges.
    # 3. For each candidate start, compute phi for a window and check distinctness.
    #
    # To speed up: use phi_fast which uses precomputed small primes.
    # For numbers up to 10^12, trial division up to 10^6 leaves a cofactor <= 10^6,
    # which is prime or 1, so we don't need Pollard's rho for phi computation.
    # Actually, after dividing by primes up to 10^6, the remaining factor is either
    # 1 or prime (since if it were composite, it would have a factor <= sqrt(10^12/10^6)=10^3,
    # which we already tried). Wait, that's not quite right.
    # If n <= 10^12 and we divide out all primes up to 10^6, the remaining cofactor
    # is either 1 or prime. Because if cofactor c > 1 is composite, it has a prime
    # factor <= sqrt(c). Since c <= 10^12 / (product of small primes removed), but actually
    # c <= 10^12. sqrt(10^12) = 10^6. So if c is composite, it has a prime factor <= 10^6,
    # which we would have already divided out. So yes, c is prime or 1.
    # Thus phi_fast with trial division up to 10^6 is sufficient and fast.

    best_k = 1
    best_s = 1

    # Phase 1: Quick scan of small range [1, 50000]
    def scan_range(start, end, window=200):
        nonlocal best_k, best_s
        if start > end:
            return
        # Compute phi for the range
        vals = []
        for x in range(start, min(end + 1, start + window + 200)):
            vals.append(phi_fast(x))
        # Now slide window
        for s in range(start, min(end, start + 200)):
            seen = set()
            k = 0
            while s + k < len(vals) and vals[k] not in seen:
                seen.add(vals[k])
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
            k += 1  # this is a rough approximation; let's do it more carefully

    # Better: direct scan
    def scan_direct(start, end):
        nonlocal best_k, best_s
        for s in range(start, min(end + 1, start + 200)):
            if now() - start_time > DEADLINE - 5:
                return
            seen = set()
            k = 0
            for x in range(s, min(s + 200, end + 1)):
                if x > 10**12:
                    break
                v = phi_fast(x)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s

    # Phase 1: Scan [1, 50000]
    scan_direct(1, 50000)

    # Phase 2: Random sampling in larger ranges
    # Try random starts in [1, 10^12] and check windows
    random.seed(42)
    attempts = 0
    while now() - start_time < DEADLINE - 10:
        # Choose a random start
        s = random.randint(1, 10**12 - 200)
        # Check window of size up to 200
        seen = set()
        k = 0
        for x in range(s, min(s + 200, 10**12 + 1)):
            v = phi_fast(x)
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            # If we found a good one, try nearby starts
            for ds in range(-50, 51):
                if now() - start_time > DEADLINE - 5:
                    break
                s2 = s + ds
                if s2 < 1 or s2 + 200 > 10**12 + 1:
                    continue
                seen2 = set()
                k2 = 0
                for x in range(s2, min(s2 + 200, 10**12 + 1)):
                    v = phi_fast(x)
                    if v in seen2:
                        break
                    seen2.add(v)
                    k2 += 1
                if k2 > best_k:
                    best_k = k2
                    best_s = s2
        attempts += 1

    # Phase 3: Targeted search near known good structures
    # Numbers with many small prime factors tend to have smaller phi values,
    # but we want distinct phi values. Let's try around highly composite numbers.
    # Also try near primes (phi(p) = p-1, which is unique for each prime).
    
    # Try starts near multiples of large primes
    for p in [10**6, 10**7, 10**8, 10**9, 10**10]:
        if now() - start_time > DEADLINE - 5:
            break
        for offset in range(-100, 101):
            if now() - start_time > DEADLINE - 5:
                break
            s = p + offset
            if s < 1 or s + 200 > 10**12 + 1:
                continue
            seen = set()
            k = 0
            for x in range(s, min(s + 200, 10**12 + 1)):
                v = phi_fast(x)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s

    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
