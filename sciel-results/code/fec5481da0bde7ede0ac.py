"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def now():
        return time.time() - start_time

    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Small primes check
        for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as 2^r * d
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Test with deterministic bases for n < 3,317,044,064,679,887,385,961,981
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

    # Precomputed small primes for trial division
    SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
    SMALL_PRIMES_SET = set(SMALL_PRIMES)
    
    # Pollard's rho for factoring
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        # Trial division by small primes first
        for p in SMALL_PRIMES:
            if p * p > n:
                break
            if n % p == 0:
                return p
        while True:
            x = random.randint(2, n - 1)
            y = x
            d = 1
            c = random.randint(1, n - 1)
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factor(n):
        """Return list of prime factors (with multiplicity) of n."""
        if n <= 1:
            return []
        factors = []
        # Check small primes first
        for p in SMALL_PRIMES:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        # Use Pollard's rho recursively
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
                continue
            d = pollard_rho(m)
            if d == m:
                # Retry
                stack.append(m)
                continue
            stack.append(d)
            stack.append(m // d)
        return factors

    # Cache for phi values to avoid recomputation
    _phi_cache = {}
    
    def phi(x):
        """Compute Euler's totient function phi(x) with exact factorization."""
        if x in _phi_cache:
            return _phi_cache[x]
        if x <= 1:
            result = 1 if x == 1 else 0
            _phi_cache[x] = result
            return result
        factors = factor(x)
        # Get unique prime factors
        unique_primes = set(factors)
        result = x
        for p in unique_primes:
            result -= result // p
        _phi_cache[x] = result
        return result

    # Phase 1: Exhaustive search in small range [1, 30000]
    N1 = 30000
    values = [0] * (N1 + 1)
    for x in range(1, N1 + 1):
        values[x] = phi(x)

    best_k = 1
    best_s = 1

    # Use a sliding window approach for efficiency
    for s in range(1, N1 - 199):
        seen = set()
        k = 0
        while s + k <= N1 and values[s + k] not in seen:
            seen.add(values[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        if now() > DEADLINE - 60:
            break

    # Phase 2: Random sampling in larger ranges
    attempts = 0
    max_attempts = 12000

    while now() < DEADLINE - 10 and attempts < max_attempts:
        attempts += 1

        # Choose a random starting point with stratified sampling
        r = random.random()
        if r < 0.10:
            s = random.randint(1000, 100000)
        elif r < 0.25:
            s = random.randint(100000, 10000000)
        elif r < 0.45:
            s = random.randint(10000000, 1000000000)
        elif r < 0.70:
            s = random.randint(1000000000, 100000000000)
        else:
            s = random.randint(100000000000, 1000000000000)

        if s + 199 > 10**12:
            s = 10**12 - 199
        if s < 1:
            s = 1

        # Compute phi values for a window starting at s
        seen = set()
        k = 0
        max_check = min(200, 10**12 - s + 1)

        for i in range(max_check):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1

        if k > best_k:
            best_k = k
            best_s = s
            # If we found a good run, try nearby starts aggressively
            if now() < DEADLINE - 15:
                for delta in range(-50, 51):
                    if now() > DEADLINE - 10:
                        break
                    s2 = s + delta
                    if s2 < 1 or s2 + 199 > 10**12:
                        continue
                    seen2 = set()
                    k2 = 0
                    max_check2 = min(200, 10**12 - s2 + 1)
                    for i in range(max_check2):
                        val = phi(s2 + i)
                        if val in seen2:
                            break
                        seen2.add(val)
                        k2 += 1
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2
        elif k >= best_k - 5:
            # Near-miss: check a few nearby starts
            if now() < DEADLINE - 15:
                for delta in [1, -1, 2, -2, 3, -3]:
                    if now() > DEADLINE - 10:
                        break
                    s2 = s + delta
                    if s2 < 1 or s2 + 199 > 10**12:
                        continue
                    seen2 = set()
                    k2 = 0
                    max_check2 = min(200, 10**12 - s2 + 1)
                    for i in range(max_check2):
                        val = phi(s2 + i)
                        if val in seen2:
                            break
                        seen2.add(val)
                        k2 += 1
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2
                        break

    # Phase 3: Local optimization around best found start
    if best_k > 1 and now() < DEADLINE - 15:
        # First pass: check a wider range
        for delta in range(-100, 101):
            if now() > DEADLINE - 10:
                break
            s2 = best_s + delta
            if s2 < 1 or s2 + 199 > 10**12:
                continue
            seen = set()
            k = 0
            max_check = min(200, 10**12 - s2 + 1)
            for i in range(max_check):
                val = phi(s2 + i)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s2

    # Phase 4: Final targeted search - look for runs that break just after best_k
    if best_k >= 10 and now() < DEADLINE - 20:
        # Search for starts where phi values are distinct for best_k+1 terms
        # by checking around the best start more carefully
        for delta in range(-200, 201):
            if now() > DEADLINE - 10:
                break
            s2 = best_s + delta
            if s2 < 1 or s2 + 199 > 10**12:
                continue
            seen = set()
            k = 0
            max_check = min(200, 10**12 - s2 + 1)
            for i in range(max_check):
                val = phi(s2 + i)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s2

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
