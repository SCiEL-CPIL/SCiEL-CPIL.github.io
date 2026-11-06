"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def now():
        return time.time() - start_time

    def segmented_phi(lo, hi):
        """Compute phi(n) for all n in [lo, hi] using segmented sieve with NumPy."""
        n = hi - lo + 1
        # phi_arr[i] = phi(lo + i), initialized to lo + i
        phi_arr = np.arange(lo, hi + 1, dtype=np.int64)
        
        # Generate primes up to sqrt(hi)
        limit = int(math.isqrt(hi)) + 1
        if limit >= 2:
            is_prime_small = np.ones(limit + 1, dtype=bool)
            is_prime_small[0] = is_prime_small[1] = False
            for i in range(2, int(math.isqrt(limit)) + 1):
                if is_prime_small[i]:
                    is_prime_small[i*i::i] = False
            primes = np.where(is_prime_small)[0]
            
            for p in primes:
                if p > limit:
                    break
                # Find first multiple of p in [lo, hi]
                first = ((lo + p - 1) // p) * p
                if first > hi:
                    continue
                rel_start = first - lo
                indices = np.arange(rel_start, n, p)
                vals = phi_arr[indices]
                phi_arr[indices] = vals - vals // p
        
        return phi_arr

    def longest_distinct_run(phi_arr):
        """Find the longest run of consecutive distinct values in phi_arr.
        Returns (best_k, best_offset) where offset is 0-indexed within phi_arr."""
        if len(phi_arr) == 0:
            return 0, 0
        
        best_k = 1
        best_offset = 0
        current_k = 1
        seen = {phi_arr[0]}
        
        for i in range(1, len(phi_arr)):
            if phi_arr[i] in seen:
                if current_k > best_k:
                    best_k = current_k
                    best_offset = i - current_k
                # Reset
                seen = {phi_arr[i]}
                current_k = 1
            else:
                seen.add(phi_arr[i])
                current_k += 1
                if current_k > best_k:
                    best_k = current_k
                    best_offset = i - current_k + 1
        
        if current_k > best_k:
            best_k = current_k
            best_offset = len(phi_arr) - current_k
        
        return best_k, best_offset

    # Phase 1: Exhaustive search in small range [1, 100000]
    N1 = 100000
    phi_small = segmented_phi(1, N1)
    best_k, best_offset = longest_distinct_run(phi_small)
    best_s = best_offset + 1  # 1-indexed
    
    if now() > DEADLINE - 30:
        return {'start': best_s, 'length': best_k}

    # Phase 2: Search in medium range [100000, 5000000] with segments
    SEG_SIZE = 50000
    lo = 100000
    hi_max = 5000000
    while lo < hi_max and now() < DEADLINE - 20:
        hi = min(lo + SEG_SIZE - 1, hi_max)
        phi_seg = segmented_phi(lo, hi)
        k, offset = longest_distinct_run(phi_seg)
        if k > best_k:
            best_k = k
            best_s = lo + offset
        lo = hi + 1

    if now() > DEADLINE - 10:
        return {'start': best_s, 'length': best_k}

    # Phase 3: Random sampling in large ranges
    SEG_SIZE_R = 200000
    max_attempts = 500
    attempts = 0
    while now() < DEADLINE - 5 and attempts < max_attempts:
        attempts += 1
        
        r = random.random()
        if r < 0.2:
            s = random.randint(10**6, 10**7)
        elif r < 0.4:
            s = random.randint(10**7, 10**8)
        elif r < 0.6:
            s = random.randint(10**8, 10**9)
        elif r < 0.8:
            s = random.randint(10**9, 10**10)
        else:
            s = random.randint(10**10, 10**11)
        
        s = max(1, s)
        hi = min(s + SEG_SIZE_R - 1, 10**12)
        if hi - s + 1 < 100:
            continue
        
        phi_seg = segmented_phi(s, hi)
        k, offset = longest_distinct_run(phi_seg)
        if k > best_k:
            best_k = k
            best_s = s + offset

    # Phase 4: Local optimization around best found start
    if now() < DEADLINE - 2:
        for delta in range(-50, 51):
            if now() > DEADLINE - 1:
                break
            s2 = best_s + delta
            if s2 < 1:
                continue
            hi2 = min(s2 + 300, 10**12)
            if hi2 - s2 + 1 < 1:
                continue
            phi_seg = segmented_phi(s2, hi2)
            k, offset = longest_distinct_run(phi_seg)
            if k > best_k:
                best_k = k
                best_s = s2 + offset

    # Ensure constraints
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, int(10**12 - best_s + 1))
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
