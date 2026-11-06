"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def sieve_primes(limit):
        """Generate primes up to limit using sieve."""
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute primes up to 1e6 for fast factorization
    primes = sieve_primes(1000000)
    
    def phi_fast(n):
        """Compute phi(n) using precomputed primes."""
        if n <= 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result
    
    def check_run(start, max_length=200):
        """Check if phi values from start are pairwise distinct for up to max_length."""
        seen = set()
        k = 0
        for i in range(max_length):
            if time.time() > deadline - 5:
                break
            n = start + i
            if n > 10**12:
                break
            v = phi_fast(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k
    
    best_k = 1
    best_start = 1
    
    # Strategy 1: Dense scan with quick rejection
    # Only do full check if first 10 values are distinct
    for s in range(1, 20000):
        if time.time() > deadline - 30:
            break
        # Quick check: first 10 phi values must be distinct
        quick_vals = set()
        quick_ok = True
        for i in range(10):
            v = phi_fast(s + i)
            if v in quick_vals:
                quick_ok = False
                break
            quick_vals.add(v)
        
        if quick_ok:
            k = check_run(s, 200)
            if k > best_k:
                best_k = k
                best_start = s
                # Explore neighborhood of good start
                for ds in range(-30, 31):
                    if time.time() > deadline - 30:
                        break
                    ns = s + ds
                    if ns < 1 or ns + 200 > 10**12:
                        continue
                    nk = check_run(ns, 200)
                    if nk > best_k:
                        best_k = nk
                        best_start = ns
    
    # Strategy 2: Random search in various ranges with quick rejection
    search_ranges = [
        (10**4, 10**6),
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9),
        (10**9, 10**10),
        (10**10, 10**11),
        (10**11, 10**12 - 200),
    ]
    
    for lo, hi in search_ranges:
        if time.time() > deadline - 10:
            break
        n_samples = 100
        for _ in range(n_samples):
            if time.time() > deadline - 10:
                break
            s = random.randint(lo, max(lo, hi - 200))
            
            # Quick check: first 10 phi values must be distinct
            quick_vals = set()
            quick_ok = True
            for i in range(10):
                v = phi_fast(s + i)
                if v in quick_vals:
                    quick_ok = False
                    break
                quick_vals.add(v)
            
            if quick_ok:
                k = check_run(s, 200)
                if k > best_k:
                    best_k = k
                    best_start = s
                    # Explore neighborhood
                    for ds in range(-50, 51):
                        if time.time() > deadline - 10:
                            break
                        ns = s + ds
                        if ns < 1 or ns + 200 > 10**12:
                            continue
                        nk = check_run(ns, 200)
                        if nk > best_k:
                            best_k = nk
                            best_start = ns
    
    # Strategy 3: Targeted search near where phi values might be distinct
    if time.time() < deadline - 5:
        # Try specific promising ranges based on number theory insights
        # Numbers with many distinct prime factors tend to have more distinct phi values
        promising_starts = [
            2**20, 2**21, 2**22, 2**23, 2**24, 2**25, 2**26, 2**27, 2**28,
            10**5, 10**6, 10**7, 10**8, 10**9, 10**10, 10**11,
            999999, 9999999, 99999999, 999999999, 9999999999,
            3628800, 39916800, 6227020800, 1307674368000,
        ]
        for s in promising_starts:
            if time.time() > deadline - 5:
                break
            if s + 200 > 10**12:
                continue
            k = check_run(s, 200)
            if k > best_k:
                best_k = k
                best_start = s
    
    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
