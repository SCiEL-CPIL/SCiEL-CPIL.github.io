"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 148.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi] using optimized segmented sieve."""
        n = hi - lo + 1
        if n <= 0:
            return []
        
        if lo == 1:
            result = [1]
            lo = 2
            n = hi - lo + 1
            if n <= 0:
                return result
        
        limit = int(math.isqrt(hi)) + 1
        
        # Sieve primes up to limit
        is_prime = bytearray(b'\x01') * (limit + 1)
        if limit >= 0:
            is_prime[0] = 0
        if limit >= 1:
            is_prime[1] = 0
        primes = []
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
                if i * i <= limit:
                    for j in range(i * i, limit + 1, i):
                        is_prime[j] = 0
        
        # phi[i] = phi(lo + i), rem[i] = remaining part after removing small prime factors
        phi = list(range(lo, hi + 1))
        rem = list(range(lo, hi + 1))
        
        for p in primes:
            # Find first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            if start < lo:
                start += p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if rem[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        # Handle remaining prime factors (> limit)
        for i in range(n):
            if rem[i] > 1:
                phi[i] -= phi[i] // rem[i]
        
        if lo == 2:
            return phi
        else:
            return result + phi
    
    def find_longest_distinct_run(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values using sliding window."""
        n = len(phi_arr)
        if n == 0:
            return offset, 1
        
        best_k = 1
        best_s = offset
        left = 0
        seen = {}
        
        for right in range(n):
            val = phi_arr[right]
            seen[val] = seen.get(val, 0) + 1
            
            while seen[val] > 1:
                left_val = phi_arr[left]
                seen[left_val] -= 1
                if seen[left_val] == 0:
                    del seen[left_val]
                left += 1
            
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = offset + left
            
            if best_k >= max_len:
                break
        
        if best_k > max_len:
            best_k = max_len
            best_s = offset + (right - best_k + 1)
        
        return best_s, best_k
    
    best_s = 1
    best_k = 1
    
    # Strategy: Search in ranges where consecutive numbers have many distinct prime factors
    # This makes phi values more likely to be distinct
    
    # Precompute some good starting points by looking at primorials and similar structures
    # Numbers near products of small primes tend to have varied phi values
    
    # Generate candidate ranges based on heuristics
    # Large ranges near 10^12, 10^11, etc.
    
    # First, do a quick scan of moderate ranges to find good starting points
    # Then refine around those points
    
    # Phase 1: Scan several ranges at different scales
    scan_ranges = [
        (10**12 - 2000000, 10**12),
        (10**12 - 5000000, 10**12 - 3000000),
        (10**11, 10**11 + 1000000),
        (10**11 - 1000000, 10**11),
        (7 * 10**11, 7 * 10**11 + 1000000),
        (3 * 10**11, 3 * 10**11 + 1000000),
        (10**10, 10**10 + 1000000),
        (10**10 - 1000000, 10**10),
        (5 * 10**9, 5 * 10**9 + 1000000),
        (10**9, 10**9 + 1000000),
        (10**9 - 1000000, 10**9),
        (10**8, 10**8 + 1000000),
        (10**7, 10**7 + 1000000),
        (10**6, 10**6 + 1000000),
    ]
    
    # Phase 2: After finding good runs, try to extend them
    
    for lo, hi in scan_ranges:
        if time_left() < 10.0:
            break
        
        if hi > 10**12:
            hi = 10**12
        if lo < 1:
            lo = 1
        if lo > hi:
            continue
        
        # Limit segment size for efficiency
        max_seg = 1500000
        if hi - lo + 1 > max_seg:
            hi = lo + max_seg - 1
        
        try:
            phi_arr = compute_phi_segmented(lo, hi)
            if time_left() < 5.0:
                break
            
            s, k = find_longest_distinct_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
                
            if best_k >= 180:
                break
        except Exception:
            continue
    
    # Phase 3: If we found a decent run, try to extend it in both directions
    if best_k >= 10 and time_left() > 20.0:
        # Try extending the best run found
        extend_lo = max(1, best_s - 50000)
        extend_hi = min(10**12, best_s + best_k - 1 + 50000)
        
        if extend_hi - extend_lo + 1 <= 2000000:
            try:
                phi_arr = compute_phi_segmented(extend_lo, extend_hi)
                if time_left() > 5.0:
                    s, k = find_longest_distinct_run(phi_arr, extend_lo, max_len=200)
                    if k > best_k:
                        best_k = k
                        best_s = s
            except Exception:
                pass
    
    # Phase 4: Try some additional targeted searches
    if best_k < 100 and time_left() > 15.0:
        # Try ranges around numbers with many small prime factors
        extra_ranges = [
            (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23, 
             2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 + 200000),
            (10**12 - 100000, 10**12),
            (999999999999 - 100000, 999999999999),
        ]
        
        for lo, hi in extra_ranges:
            if time_left() < 8.0:
                break
            if hi > 10**12:
                hi = 10**12
            if lo < 1:
                lo = 1
            if lo > hi:
                continue
            if hi - lo + 1 > 500000:
                hi = lo + 500000 - 1
            
            try:
                phi_arr = compute_phi_segmented(lo, hi)
                if time_left() < 5.0:
                    break
                s, k = find_longest_distinct_run(phi_arr, lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Final safety checks
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
        best_k = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
