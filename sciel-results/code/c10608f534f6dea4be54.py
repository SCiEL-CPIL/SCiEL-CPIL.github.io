import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    # Sieve primes up to sqrt(10^12) = 10^6
    limit = 10**6 + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    best_k = 1
    best_s = 1
    
    # Use smaller block size for better time management
    W = 2 * 10**6
    
    max_val = 10**12
    
    # Strategy: search in ranges where long distinct-totient runs are more likely.
    # Numbers with many distinct prime factors tend to have more varied totients.
    # Focus on ranges near multiples of primorials or in mid-range values.
    
    # Generate search ranges with strategic placement
    search_ranges = []
    
    # High-value ranges: scan from top down in 2*10^10 chunks
    step = 2 * 10**10
    for i in range(50):
        hi = max_val - i * step
        lo = max(1, hi - step + 1)
        if lo >= hi:
            break
        search_ranges.append((lo, hi))
    
    # Deduplicate and sort by priority (higher ranges first)
    seen = set()
    unique_ranges = []
    for lo, hi in search_ranges:
        key = (lo, hi)
        if key not in seen and lo < hi:
            seen.add(key)
            unique_ranges.append((lo, hi))
    
    # Sort: prioritize higher start values
    unique_ranges.sort(key=lambda x: -x[0])
    
    for lo, hi in unique_ranges:
        if time_left() < 12.0:
            break
            
        # Process this range in chunks
        current_lo = lo
        while current_lo <= hi:
            if time_left() < 8.0:
                break
                
            current_hi = min(current_lo + W - 1, hi)
            n = current_hi - current_lo + 1
            
            # Initialize phi and rem
            phi = np.arange(current_lo, current_hi + 1, dtype=np.int64)
            rem = np.arange(current_lo, current_hi + 1, dtype=np.int64)
            
            # Process primes
            max_p = int(math.isqrt(current_hi))
            idx_end = np.searchsorted(primes, max_p + 1)
            relevant_primes = primes[:idx_end]
            
            for p in relevant_primes:
                p = int(p)
                start = ((current_lo + p - 1) // p) * p
                if start > current_hi:
                    continue
                    
                indices = np.arange(start, current_hi + 1, p) - current_lo
                phi[indices] -= phi[indices] // p
                
                # Divide out all powers of p from rem
                r = rem[indices].copy()
                while True:
                    m = r % p == 0
                    if not np.any(m):
                        break
                    r[m] //= p
                rem[indices] = r
            
            # For numbers where rem > 1, there's a remaining prime factor
            mask_rem = rem > 1
            if np.any(mask_rem):
                phi[mask_rem] -= phi[mask_rem] // rem[mask_rem]
            
            # Find longest run of distinct values
            last_seen = {}
            start_idx = 0
            for i in range(len(phi)):
                v = phi[i]
                if v in last_seen:
                    if last_seen[v] >= start_idx:
                        start_idx = last_seen[v] + 1
                last_seen[v] = i
                k = i - start_idx + 1
                if k > best_k:
                    best_k = k
                    best_s = current_lo + start_idx
                    if best_k >= 200:
                        return {'start': best_s, 'length': best_k}
                    
            current_lo = current_hi + 1
            
    return {'start': best_s, 'length': best_k}
