"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    _prime_cache = {}
    
    def sieve_primes(limit):
        """Sieve primes up to limit using bytearray, with caching."""
        if limit in _prime_cache:
            return _prime_cache[limit]
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                start = i * i
                step = i
                is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        result = [i for i in range(2, limit + 1) if is_prime[i]]
        _prime_cache[limit] = result
        return result
    
    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi] using segmented sieve with NumPy."""
        n = hi - lo + 1
        if n <= 0:
            return []
        limit = int(math.isqrt(hi)) + 1
        primes = sieve_primes(limit)
        
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        remaining = np.arange(lo, hi + 1, dtype=np.int64)
        
        for p in primes:
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
            # Indices in [0, n) where (lo + idx) is divisible by p
            idxs = np.arange(start - lo, n, p)
            if len(idxs) == 0:
                continue
            
            # Update phi for all multiples of p
            # phi[x] = phi[x] * (1 - 1/p)
            phi[idxs] = phi[idxs] // p * (p - 1)
            
            # Remove p from remaining
            rem_sel = remaining[idxs].copy()
            # Divide by p repeatedly using fixed iterations for speed/predictability
            # Max power for 10^12 is ~40 for p=2
            for _ in range(50):
                mask = rem_sel % p == 0
                if not mask.any():
                    break
                rem_sel[mask] = rem_sel[mask] // p
            remaining[idxs] = rem_sel
        
        # Handle remaining prime factors
        mask = remaining > 1
        if np.any(mask):
            phi[mask] = phi[mask] // remaining[mask] * (remaining[mask] - 1)
        
        return phi.tolist()
    
    def find_best_run(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values in phi_arr using sliding window."""
        n = len(phi_arr)
        if n == 0:
            return offset, 0
        best_k = 0
        best_s = offset
        left = 0
        seen = {}
        
        for right in range(n):
            val = phi_arr[right]
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = right
            current_len = right - left + 1
            if current_len > best_k:
                best_k = min(current_len, max_len)
                best_s = offset + left
            if best_k >= max_len:
                break
        
        return best_s, best_k
    
    best_s, best_k = 1, 1
    
    # Strategy: Random sampling of segments across the range [1, 10^12].
    # This allows us to probe deep into the range where longer runs might exist,
    # avoiding the slow linear scan that timed out.
    
    import random
    random.seed(12345)
    
    SEGMENT_SIZE = 50000
    max_n = 10**12
    
    # Pre-sieve primes up to 10^6 once to speed up all segments
    # 10^6 is sqrt(10^12), sufficient for the max range
    sieve_primes(10**6)
    
    while time_left() > 10.0:
        # Pick a random start point
        # Ensure s + SEGMENT_SIZE - 1 <= max_n
        max_start = max_n - SEGMENT_SIZE
        if max_start < 1:
            break
        s = random.randint(1, max_start)
        hi = s + SEGMENT_SIZE - 1
        
        try:
            phi_arr = compute_phi_segmented(s, hi)
            run_s, run_k = find_best_run(phi_arr, s, max_len=200)
            if run_k > best_k:
                best_k = run_k
                best_s = run_s
        except Exception:
            pass
    
    # Refinement around best found run
    if best_k >= 2 and time_left() > 15.0:
        # Expand search window around best_s
        for delta in [-5000, -2500, -1000, 1000, 2500, 5000]:
            if time_left() < 5.0:
                break
            extend_lo = max(1, best_s + delta - 1000)
            extend_hi = min(10**12, best_s + best_k + 1000 + delta)
            if extend_hi - extend_lo < 100:
                continue
            try:
                phi_arr = compute_phi_segmented(extend_lo, extend_hi)
                s, k = find_best_run(phi_arr, extend_lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
