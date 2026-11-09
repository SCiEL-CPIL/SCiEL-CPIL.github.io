import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 150s limit

    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """Compute phi for a segment [start, end] using segmented sieve approach."""
        size = end - start + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize phi and remainder arrays
        vals = start + np.arange(size, dtype=np.int64)
        phi = vals.copy()
        rem = vals.copy()
        
        # Process primes
        # We only need primes up to sqrt(end) to ensure remaining factors are prime
        # sqrt(10^12) = 10^6
        for p in primes:
            if p * p > end:
                break
            # Find indices where rem is divisible by p
            mask = (rem % p) == 0
            if not mask.any():
                continue
            # Update phi: phi[n] -= phi[n] / p
            phi[mask] -= phi[mask] // p
            # Remove all factors of p from rem
            rem[mask] //= p
            while True:
                still_div = (rem[mask] % p) == 0
                if not still_div.any():
                    break
                # Get indices that are still divisible
                idxs = np.where(mask)[0]
                sub_idx = idxs[still_div]
                rem[sub_idx] //= p
                # Note: we do NOT update phi again, as phi factorization 
                # only requires subtracting once per distinct prime factor.
            
        # Handle remaining prime factors > sqrt(end)
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    def find_longest_run(phi_vals):
        """Find longest run of distinct phi values using sliding window."""
        n = len(phi_vals)
        if n <= 1:
            return 1, 0
        
        best_k = 1
        best_s = 0
        seen = {}
        left = 0
        
        for right in range(n):
            val = int(phi_vals[right])
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = left
            if best_k >= 200:
                break
                
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Search strategy:
    # Larger numbers have larger phi values and are less likely to collide?
    # Actually, phi(n) is roughly n. The "density" of phi values is lower for large n?
    # Collisions like phi(n) = phi(n+1) are rare.
    # We should search near the upper bound 10^12.
    
    seg_size = 50000
    # We will scan downwards from 10^12
    # To avoid missing a run that crosses segment boundaries, 
    # we can overlap segments or just be careful.
    # Given k<=200 and seg_size=50000, overlap is not strictly necessary 
    # if we just look for max k in each segment, but a run could be split.
    # However, if we find a run of length k in a segment, it's a valid run.
    # If a longer run exists, it's likely to be fully contained in a segment 
    # unless it's exactly at the boundary.
    # We can just scan segments.
    
    current_end = 10**12
    while current_end > 1 and time.time() - start_time < deadline - 5:
        seg_start = max(1, current_end - seg_size + 1)
        seg_end = current_end
        
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s_offset = find_longest_run(phi_vals)
            actual_s = seg_start + s_offset
            if k > best_k:
                best_k = k
                best_s = actual_s
                if best_k >= 200:
                    break
        except Exception:
            pass
        
        # Move to next segment
        # Overlap by 200 to be safe? Or just step by seg_size?
        # If we step by seg_size, we might miss a run that spans the boundary.
        # But a run of length 200 is small compared to 50000.
        # The probability of a run of length 200 being split by a boundary is low.
        # To be safe, we can overlap by 200.
        current_end = seg_start - 200
        
    # Final validation
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = max(1, 10**12 - best_s + 1)

    return {'start': best_s, 'length': best_k}
