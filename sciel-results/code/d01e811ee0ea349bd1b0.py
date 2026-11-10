"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 using sieve
    # sqrt(10^12) = 10^6, so primes up to 10^6 are sufficient for factorization
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)

    def phi_fast(n):
        if n == 1:
            return 1
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    def check_run(s, max_k=200):
        """Check if phi(s), ..., phi(s+max_k-1) are distinct. Returns length of distinct prefix."""
        seen = set()
        k = 0
        for i in range(max_k):
            v = phi_fast(s + i)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Strategy: For large n, phi(n) values are distinct with very high probability.
    # We target a range near 10^12 to maximize the chance of finding k=200.
    # We verify a specific block. If it fails (collision), we shift slightly.
    
    # Start near the upper limit to ensure s + k - 1 <= 10^12
    # Let's try s = 10^12 - 200
    base_s = 10**12 - 200
    
    best_k = 1
    best_s = 1

    # Try a few candidate starts near the top
    # We only need to find one valid run of length 200.
    # Given the rarity of collisions for large n, the first try should likely succeed.
    
    for offset in range(0, 10):
        if time_left() < 5:
            break
        s = base_s - offset
        if s < 1:
            continue
        
        # Check if this run has length 200
        k = check_run(s, 200)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        if best_k == 200:
            break

    # If for some reason we didn't find 200 (highly unlikely), do a small local search
    # to ensure we return a valid result, even if shorter.
    if best_k < 200 and time_left() > 10:
        # Try scanning backwards from base_s
        for s in range(base_s, base_s - 1000, -1):
            if time_left() < 5:
                break
            k = check_run(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
