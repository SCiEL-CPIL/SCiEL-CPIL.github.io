import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s safety margin
    
    # Precompute small primes for trial division
    SIEVE_LIMIT = 100000
    is_prime_small = [True] * (SIEVE_LIMIT + 1)
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if is_prime_small[i]:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                is_prime_small[j] = False
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_small[i]]
    
    # Miller-Rabin for large numbers
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < SIEVE_LIMIT + 1:
            return is_prime_small[n]
        # Check small primes first
        for p in small_primes[:50]:
            if n % p == 0:
                return n == p
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
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

    def phi(n):
        if n == 1:
            return 1
        if n < SIEVE_LIMIT + 1:
            # Direct computation using precomputed primes
            result = n
            for p in small_primes:
                if p * p > n:
                    break
                if n % p == 0:
                    result -= result // p
            if n > 1:
                result -= result // n
            return result
        # For larger n, use trial division by small primes, then Miller-Rabin
        result = n
        temp = n
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is either prime or a product of two large primes
            if is_prime_mr(temp):
                result -= result // temp
            else:
                # temp = p * q where p, q are large primes
                # We need to find them. Use a simple approach:
                # Since temp < 10^12, sqrt(temp) < 10^6
                # Try to factor temp
                sqrt_temp = int(math.isqrt(temp))
                found = False
                for p in range(small_primes[-1] + 1, sqrt_temp + 1):
                    if is_prime_mr(p) and temp % p == 0:
                        q = temp // p
                        result -= result // p
                        if q > 1 and is_prime_mr(q):
                            result -= result // q
                        found = True
                        break
                if not found:
                    # Shouldn't happen for temp < 10^12 with our sieve
                    result -= result // temp
        return result

    def check_run(s, max_k=200):
        seen = set()
        k = 0
        while k < max_k:
            val = phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    best_k = 1
    best_s = 1

    # Phase 1: Exhaustive search in small range
    # We can check s from 1 to ~500000 quickly
    limit_phase1 = 500000
    for s in range(1, min(limit_phase1 + 1, 10**12 - 200)):
        if time.time() > DEADLINE:
            break
        k = check_run(s)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Search in medium range with stride
    # Check every 100th number from 500000 to 5000000
    if time.time() < DEADLINE:
        start_phase2 = 500000
        end_phase2 = min(5000000, 10**12 - 200)
        for s in range(start_phase2, end_phase2 + 1, 100):
            if time.time() > DEADLINE:
                break
            k = check_run(s)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Phase 3: If we have time, do a more careful search around promising areas
    # Look for runs where phi values are likely to be distinct
    # Strategy: search in blocks, and if we find a run of length >= 10,
    # do a detailed search in the neighborhood
    
    if time.time() < DEADLINE:
        # Search in larger range with larger stride
        start_phase3 = 5000000
        end_phase3 = min(10**8, 10**12 - 200)
        for s in range(start_phase3, end_phase3 + 1, 1000):
            if time.time() > DEADLINE:
                break
            k = check_run(s)
            if k > best_k:
                best_k = k
                best_s = s
            # If we find a decent run, check neighbors
            if k >= max(10, best_k - 2):
                for delta in range(-5, 6):
                    if time.time() > DEADLINE:
                        break
                    s2 = s + delta
                    if 1 <= s2 and s2 + 199 <= 10**12:
                        k2 = check_run(s2)
                        if k2 > best_k:
                            best_k = k2
                            best_s = s2

    return {'start': best_s, 'length': best_k}
