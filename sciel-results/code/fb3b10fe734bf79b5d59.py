import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    # Sieve primes up to 10^6 (sqrt of 10^12)
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_segment(start, end):
        """Compute phi for all numbers in [start, end] using precomputed primes."""
        size = end - start + 1
        phi = list(range(start, end + 1))
        rem = list(range(start, end + 1))
        
        for p in primes:
            if p * p > end:
                break
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            for m in range(first, end + 1, p):
                idx = m - start
                if rem[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        for i in range(size):
            if rem[i] > 1:
                phi[i] -= phi[i] // rem[i]
        
        return phi
    
    def find_longest_run(phi_vals, seg_start, max_len=200):
        """Find the longest run of distinct phi values. Returns (length, actual_start)."""
        best_k = 1
        best_s = seg_start
        n = len(phi_vals)
        
        for s in range(n):
            if time.time() - start_time > deadline - 5:
                break
            seen = set()
            k = 0
            while s + k < n and phi_vals[s + k] not in seen:
                seen.add(phi_vals[s + k])
                k += 1
            if k > best_k:
                best_k = k
                best_s = seg_start + s
            if best_k >= max_len:
                break
        
        return best_k, best_s
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Search small ranges thoroughly [1, 200000]
    chunk_size = 50000
    search_end = 200000
    
    for seg_start in range(1, search_end + 1, chunk_size):
        seg_end = min(seg_start + chunk_size - 1, search_end)
        if time.time() - start_time > deadline - 40:
            break
        
        phi_vals = compute_phi_segment(seg_start, seg_end)
        k, s = find_longest_run(phi_vals, seg_start, 200)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Sample medium ranges [200000, 10^7]
    if time.time() - start_time < deadline - 20:
        for seg_start in range(200000, 10**7, 100000):
            if time.time() - start_time > deadline - 20:
                break
            seg_end = min(seg_start + 99999, 10**7)
            
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s = find_longest_run(phi_vals, seg_start, 200)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Phase 3: Sample large ranges
    if time.time() - start_time < deadline - 15:
        # Sample segments of size 10^6 across the range
        for seg_start in range(10**7, 10**12, 10**8):
            if time.time() - start_time > deadline - 10:
                break
            seg_end = min(seg_start + 999999, 10**12)
            
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s = find_longest_run(phi_vals, seg_start, 200)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Ensure constraints are met
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
