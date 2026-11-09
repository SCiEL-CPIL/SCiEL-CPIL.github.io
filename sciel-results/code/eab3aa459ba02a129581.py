import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    # 1. Sieve primes up to 10^6 (sqrt(10^12))
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """
        Computes phi(n) for n in [start, end] using segmented sieve.
        Returns a numpy array of phi values (int64).
        """
        size = end - start + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        
        vals = np.arange(start, end + 1, dtype=np.int64)
        phis = vals.copy()
        rems = vals.copy()
        
        # Process small prime factors
        for p in primes:
            if p * p > end:
                break
            
            # Find first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            
            # Create mask for positions where rems is divisible by p
            indices = np.arange(first_mult, end + 1, p, dtype=np.int64)
            if len(indices) == 0:
                continue
            idx_array = indices - start
            
            # Only process where rems is actually divisible by p
            mask = rems[idx_array] % p == 0
            valid_idx = idx_array[mask]
            
            if len(valid_idx) == 0:
                continue
            
            # Update phi: phi[n] -= phi[n] // p
            phis[valid_idx] -= phis[valid_idx] // p
            
            # Remove all factors of p from rems
            while True:
                still_div = rems[valid_idx] % p == 0
                if not np.any(still_div):
                    break
                valid_idx = valid_idx[still_div]
                rems[valid_idx] //= p
                    
        # Process remaining large prime factors
        large_prime_mask = rems > 1
        if np.any(large_prime_mask):
            large_indices = np.where(large_prime_mask)[0]
            p_vals = rems[large_indices]
            phis[large_indices] -= phis[large_indices] // p_vals
            
        return phis

    def find_longest_run(phi_vals):
        """
        Finds the longest run of distinct values in phi_vals.
        Returns (max_length, relative_start_index).
        """
        best_k = 1
        best_s_rel = 0
        n = len(phi_vals)
        
        if n == 0:
            return 0, 0

        # Use a dictionary to track last seen index of each value
        last_seen = {}
        current_len = 0
        current_start_rel = 0
        
        for i in range(n):
            val = phi_vals[i]
            
            if val in last_seen:
                last_idx = last_seen[val]
                if last_idx >= current_start_rel:
                    current_start_rel = last_idx + 1
            
            last_seen[val] = i
            current_len = i - current_start_rel + 1
            
            if current_len > best_k:
                best_k = current_len
                best_s_rel = current_start_rel
                
            if best_k >= 200:
                return best_k, best_s_rel
                
        return best_k, best_s_rel

    best_k = 1
    best_s = 1
    
    # Strategy: Scan large contiguous regions. The key insight is that
    # for large n, phi(n) values are spread out, so collisions are rare.
    # We need to scan enough numbers to find a long run.
    
    # Use a segment size that balances computation time and coverage
    W = 200000  # 200k numbers per segment
    
    # Scan multiple ranges in the high end where n is large
    # For n ~ 10^11, phi(n) ~ n * prod(1-1/p) which is much smaller than n
    # but still spread out enough that collisions are rare
    
    scan_ranges = [
        (10**11, 10**11 + W * 10),
        (5 * 10**11, 5 * 10**11 + W * 10),
        (9 * 10**11, 10**12),
        (2 * 10**11, 2 * 10**11 + W * 8),
        (7 * 10**11, 7 * 10**11 + W * 8),
    ]
    
    for range_start, range_end in scan_ranges:
        if time.time() - start_time > deadline:
            break
        
        for seg_start in range(range_start, range_end, W):
            if time.time() - start_time > deadline:
                break
            
            seg_end = min(seg_start + W - 1, range_end, 10**12)
            if seg_start > seg_end:
                break
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                    if best_k >= 200:
                        return {'start': best_s, 'length': best_k}
                        
            except Exception:
                continue

    # If we still have time, try random spots in unexplored ranges
    if time.time() - start_time < deadline - 10:
        random.seed(12345)
        
        for _ in range(20):
            if time.time() - start_time > deadline - 5:
                break
            
            seg_start = random.randint(10**10, 10**12 - W)
            seg_end = seg_start + W - 1
            
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s_rel = find_longest_run(phi_vals)
                actual_s = seg_start + s_rel
                
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    
                    if best_k >= 200:
                        break
                        
            except Exception:
                continue

    # Ensure constraints
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
