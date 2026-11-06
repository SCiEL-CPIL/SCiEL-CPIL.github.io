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
    
    def compute_phi_range(start, length):
        """Compute phi for range [start, start+length-1] efficiently."""
        vals = [0] * length
        for i in range(length):
            if time.time() > deadline - 5:
                break
            vals[i] = phi_fast(start + i)
        return vals
    
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
    
    # Strategy 1: Search near known good regions
    # phi values tend to be more distinct when numbers have varied factors
    # Try ranges around powers of 2, and random large ranges
    
    # First, do a quick search in small range to establish baseline
    for s in range(1, 5000):
        if time.time() > deadline - 20:
            break
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_start = s
            # If we found a long run, try nearby starts too
            for ds in range(1, 50):
                if time.time() > deadline - 20:
                    break
                ns = s + ds
                if ns + 200 > 10**12:
                    break
                nk = check_run(ns, 200)
                if nk > best_k:
                    best_k = nk
                    best_start = ns
    
    # Strategy 2: Search in larger ranges
    # Try ranges where numbers are more "random" in their factorization
    search_ranges = [
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
        # Sample several starting points in this range
        n_samples = 20
        for _ in range(n_samples):
            if time.time() > deadline - 10:
                break
            s = random.randint(lo, max(lo, hi - 200))
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
    # Numbers that are products of distinct primes often have distinct phi values
    # Try ranges with high prime density
    
    if time.time() < deadline - 5:
        # Try some specific promising ranges
        promising_starts = [
            10**5, 10**6, 10**7, 10**8, 10**9, 10**10, 10**11,
            999999, 9999999, 99999999, 999999999, 9999999999,
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
