import time
import math
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    # Precompute primes up to 10^6 using sieve
    SIEVE_LIMIT = 10**6
    is_prime_small = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_small[0] = is_prime_small[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if is_prime_small[i]:
            step = i
            start = i * i
            is_prime_small[start:SIEVE_LIMIT+1:step] = bytearray(len(range(start, SIEVE_LIMIT+1, step)))
    
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_small[i]]
    # Keep a smaller set for quick trial division
    trial_primes = [p for p in small_primes if p <= 10000]

    def is_prime_mr(n):
        """Deterministic Miller-Rabin for n < 2^64"""
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        # Small prime checks
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witnesses deterministic for 64-bit
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            # Limit iterations to prevent infinite loops on primes (shouldn't happen if is_prime works)
            for _ in range(10000):
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
                if d != 1:
                    break
            if d != n:
                return d

    def factorize(n):
        """Return list of prime factors (with multiplicity)"""
        if n == 1:
            return []
        # Trial division by small primes
        for p in trial_primes:
            if p * p > n:
                break
            if n % p == 0:
                factors = []
                while n % p == 0:
                    factors.append(p)
                    n //= p
                if n > 1:
                    factors.extend(factorize(n))
                return factors
        if is_prime_mr(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    # Cache for phi values to avoid recomputation
    phi_cache = {}

    def phi(n):
        if n in phi_cache:
            return phi_cache[n]
        if n == 1:
            phi_cache[n] = 1
            return 1
        factors = factorize(n)
        unique_factors = set(factors)
        result = n
        for p in unique_factors:
            result -= result // p
        phi_cache[n] = result
        return result

    def check_run(s, max_k=200):
        """Check how many consecutive integers starting at s have distinct totients"""
        seen = set()
        k = 0
        # Early exit if time is running out
        if time.time() > DEADLINE - 5:
            return 0
        while k < max_k:
            val = phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
            # Periodic time check
            if k % 10 == 0 and time.time() > DEADLINE - 5:
                break
        return k

    best_k = 1
    best_s = 1

    # Strategy 1: Deterministic scan of small numbers
    # Small numbers tend to have more "structure" and might yield longer runs?
    # Or at least it's fast.
    # Let's scan up to 100,000.
    # Cost: ~100k phi computations. Each phi is fast for small n.
    limit_small = 200000
    for s in range(1, limit_small + 1):
        if time.time() > DEADLINE - 20:
            break
        k = check_run(s)
        if k > best_k:
            best_k = k
            best_s = s

    # Strategy 2: Targeted Random Search
    # We want to find runs where phi(n) values are distinct.
    # Collisions are more likely if phi(n) values are similar in magnitude.
    # phi(n) is roughly n * e^-gamma * log(log(n)) on average? No, that's for distinct values in a set.
    # phi(n) < n.
    # If we pick random s, the probability of a collision in k steps is roughly k^2 / (2 * range_of_phi).
    # Range of phi is roughly [s, 2s]. So probability of no collision is exp(-k^2 / (4s)).
    # This suggests that for large s, it's easier to have distinct values?
    # But we are bounded by 10^12.
    
    # Let's do random sampling across the whole range, but biased towards larger numbers?
    # Or just uniform.
    
    # Also, let's try to "extend" found runs.
    # If we find a run of length k at s, check s-1 and s+k.
    
    random.seed(12345)
    
    # We will spend the remaining time here.
    # Each check_run takes ~k * cost(phi).
    # For large s, cost(phi) is ~1ms?
    # If k is small (5), it's ~5ms.
    # If k is large (20), it's ~20ms.
    # We have ~100s left.
    # We can do ~5000-10000 checks.
    
    # To maximize k, we should focus on finding *any* run that is longer than best_k.
    # If best_k is small, we can afford to check more candidates.
    
    # Let's use a mix of:
    # 1. Random s in [10^6, 10^12]
    # 2. Random s in [10^12 - 10^6, 10^12] (boundary effects?)
    
    while time.time() < DEADLINE - 5:
        # Pick a random start
        # Uniform in [10^6, 10^12 - 200]
        s = random.randint(10**6, 10**12 - 200)
        
        # Check this run
        k = check_run(s)
        
        if k > best_k:
            best_k = k
            best_s = s
            
            # If we found a new best, try to extend it
            # Check s-1
            if s > 1:
                k2 = check_run(s - 1)
                if k2 > best_k:
                    best_k = k2
                    best_s = s - 1
            # Check s+k
            s_end = s + k
            if s_end + 199 <= 10**12:
                k3 = check_run(s_end)
                if k3 > best_k:
                    best_k = k3
                    best_s = s_end

        # Optional: If k is close to best_k, check neighbors
        elif k >= best_k - 3 and best_k > 1:
            for delta in [-1, 1]:
                if time.time() > DEADLINE - 5:
                    break
                s2 = s + delta
                if 1 <= s2 and s2 + 199 <= 10**12:
                    k2 = check_run(s2)
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2

    return {'start': best_s, 'length': best_k}
