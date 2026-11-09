import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        size = end - start + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        phi = np.arange(start, end + 1, dtype=np.int64)
        rem = phi.copy()
        
        for p in primes:
            if p * p > end:
                break
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            idxs = np.arange(first - start, size, p)
            if len(idxs) == 0:
                continue
            
            # Update phi
            phi[idxs] = phi[idxs] // p * (p - 1)
            
            # Update rem
            rem[idxs] //= p
            # Check for further factors
            mask = rem[idxs] % p == 0
            while mask.any():
                rem[idxs[mask]] //= p
                mask = rem[idxs[mask]] % p == 0

        mask = rem > 1
        if mask.any():
            phi[mask] = phi[mask] // rem[mask] * (rem[mask] - 1)
        
        return phi

    def find_longest_run_sliding(phi_vals):
        n = len(phi_vals)
        if n == 0:
            return 1, 0
        best_k = 1
        best_s = 0
        seen = {}
        left = 0
        for right in range(n):
            val = phi_vals[right]
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            cur_k = right - left + 1
            if cur_k > best_k:
                best_k = cur_k
                best_s = left
            if best_k >= 200:
                break
        return best_k, best_s

    best_k = 1
    best_s = 1
    
    # Scan from high to low
    # Step size 500k, overlap 1000
    step = 500000
    overlap = 1000
    
    # Start near 10^12
    # We need s + k - 1 <= 10^12
    # So max start is 10^12 - 199
    # We scan segments [end - step + 1, end]
    
    current_end = 10**12
    while current_end >= 1:
        if time.time() - start_time > deadline:
            break
            
        seg_end = current_end
        seg_start = max(1, seg_end - step + 1)
        
        # Ensure we don't go below 1
        if seg_start > 10**12 - 199:
             seg_start = 10**12 - 199
             
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s_offset = find_longest_run_sliding(phi_vals)
            actual_s = seg_start + s_offset
            
            # Check if this run is valid (within bounds)
            if actual_s + k - 1 <= 10**12 and actual_s >= 1:
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    if best_k >= 200:
                        break
        except Exception:
            pass
            
        # Move to next segment
        # We want to cover the gap.
        # Next end should be seg_start - overlap
        current_end = seg_start - overlap

    # Final validation
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': int(best_s), 'length': int(best_k)}
