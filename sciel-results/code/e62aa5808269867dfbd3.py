import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # Safety margin
    
    # Miller-Rabin Primality Test
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime_mr(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in MR_BASES:
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
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        if n == 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    def phi_single(n):
        if n == 1:
            return 1
        factors = factorize(n)
        unique_factors = set(factors)
        result = n
        for p in unique_factors:
            result -= result // p
        return result

    # Segmented sieve for computing phi(n) for n in [low, high]
    def segmented_phi(low, high):
        """Compute phi(n) for all n in [low, high] using segmented sieve."""
        length = high - low + 1
        if length <= 0:
            return []
        
        # Initialize: phi[i] = low + i, spf[i] = 0 (no prime factor found yet)
        phi_vals = list(range(low, high + 1))
        spf = [0] * length  # smallest prime factor, 0 means not set
        
        # Sieve with small primes up to sqrt(high)
        limit = int(math.isqrt(high)) + 1
        # Generate primes up to limit using simple sieve
        is_prime_small = [True] * (limit + 1)
        is_prime_small[0] = False
        if limit >= 1:
            is_prime_small[1] = False
        for i in range(2, limit + 1):
            if is_prime_small[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime_small[j] = False
        
        small_primes = [p for p in range(2, limit + 1) if is_prime_small[p]]
        
        for p in small_primes:
            # Find first multiple of p in [low, high]
            first = ((low + p - 1) // p) * p
            for m in range(first, high + 1, p):
                idx = m - low
                if spf[idx] == 0:
                    spf[idx] = p
                    phi_vals[idx] -= phi_vals[idx] // p
                else:
                    # Already has a smaller prime factor; still need to divide out p
                    if spf[idx] != p:
                        phi_vals[idx] -= phi_vals[idx] // p
        
        # For numbers with no small prime factor, they might be prime or product of large primes
        # We need to handle remaining numbers where spf[i] == 0
        for i in range(length):
            if spf[i] == 0:
                n = low + i
                if n == 1:
                    phi_vals[i] = 1
                elif n == 2:
                    phi_vals[i] = 1
                else:
                    # n has no prime factor <= sqrt(n), so n is prime
                    # But wait: if n is composite, it must have a factor <= sqrt(n)
                    # Since we sieved all primes up to sqrt(high) >= sqrt(n), 
                    # if spf[i] == 0 and n > 1, then n is prime
                    if n > 1:
                        phi_vals[i] = n - 1
                    else:
                        phi_vals[i] = 1
        
        return phi_vals

    best_k = 1
    best_s = 1

    def check_start(s, max_k=200):
        nonlocal best_k, best_s
        # Use segmented sieve for efficiency
        if s + max_k - 1 > 10**12:
            max_k = 10**12 - s + 1
        if max_k <= 0:
            return False
        
        phi_vals = segmented_phi(s, s + max_k - 1)
        
        seen = set()
        k = 0
        for val in phi_vals:
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            return True
        return False

    # Phase 1: Deterministic scan of small numbers using segmented sieve
    # Scan in chunks for efficiency
    chunk_size = 5000
    scan_limit = 200000
    s = 1
    while s <= scan_limit:
        if time.time() > DEADLINE:
            break
        end = min(s + chunk_size - 1, scan_limit)
        # Check all starts in [s, end]
        # For each start i in [s, end], we need phi(i) through phi(i+199)
        # We can compute phi for [s, end + 199] once
        phi_end = min(end + 199, 10**12)
        if s <= phi_end:
            phi_vals = segmented_phi(s, phi_end)
            for i in range(s, end + 1):
                if time.time() > DEADLINE:
                    break
                if i + 199 > 10**12:
                    break
                idx = i - s
                seen = set()
                k = 0
                for j in range(200):
                    if idx + j >= len(phi_vals):
                        break
                    val = phi_vals[idx + j]
                    if val in seen:
                        break
                    seen.add(val)
                    k += 1
                if k > best_k:
                    best_k = k
                    best_s = i
                if best_k == 200:
                    return {'start': best_s, 'length': best_k}
        s = end + 1
    
    if best_k == 200:
        return {'start': best_s, 'length': best_k}
    
    # Phase 2: Random sampling in larger ranges using segmented sieve
    random.seed(42)
    
    # Try structured search: sample random starts
    while time.time() < DEADLINE - 15.0:
        s = random.randint(10**6, 10**12 - 200)
        
        if time.time() > DEADLINE:
            break
        
        check_start(s)
        
        if best_k == 200:
            break
        
        # If we found a moderately long run, explore neighbors
        if best_k > 30 and random.random() < 0.3:
            for delta in range(-20, 21):
                if delta == 0:
                    continue
                new_s = best_s + delta
                if new_s >= 1 and new_s + 199 <= 10**12:
                    if time.time() > DEADLINE:
                        break
                    check_start(new_s)
                    if best_k == 200:
                        break
            if best_k == 200:
                break
    
    return {'start': best_s, 'length': best_k}
