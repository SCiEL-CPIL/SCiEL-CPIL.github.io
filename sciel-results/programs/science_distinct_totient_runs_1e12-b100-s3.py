import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin

    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        size = end - start + 1
        vals = np.arange(start, end + 1, dtype=np.int64)
        phi = vals.copy()
        rem = vals.copy()
        
        for p in primes:
            if p * p > end:
                break
            mask = rem % p == 0
            if not mask.any():
                continue
            phi[mask] -= phi[mask] // p
            idxs = np.where(mask)[0]
            rem[idxs] //= p
            # Remove remaining factors of p
            still_div = rem[idxs] % p == 0
            sub_idxs = idxs[still_div]
            while sub_idxs.size > 0:
                rem[sub_idxs] //= p
                still_div = rem[sub_idxs] % p == 0
                sub_idxs = sub_idxs[still_div]
        
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    def find_longest_run(phi_vals):
        best_k = 1
        best_s = 0
        n = len(phi_vals)
        # Simple sliding window or just check from start?
        # We need the longest run of DISTINCT values.
        # Since phi values are large, collisions are rare.
        # We can just iterate and keep a set.
        # But we want the MAX length.
        # A run of length k means phi[s]...phi[s+k-1] are distinct.
        # If we find a collision, the run ends.
        # So we can just scan linearly.
        
        seen = set()
        current_len = 0
        current_start = 0
        
        for i in range(n):
            val = phi_vals[i]
            if val in seen:
                # Collision, reset
                # The run starting at current_start ended at i-1
                # New run starts at i
                seen = set()
                current_start = i
                current_len = 0
            seen.add(val)
            current_len += 1
            if current_len > best_k:
                best_k = current_len
                best_s = current_start
            # Optimization: if best_k is already very large, we can stop early?
            # But we need to find the MAX.
            # However, if best_k >= 200, we can stop.
            if best_k >= 200:
                break
                
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Phase 1: Small range [1, 100000]
    if time.time() - start_time < deadline - 10:
        phi_vals = compute_phi_segment(1, 100000)
        k, s = find_longest_run(phi_vals)
        if k > best_k:
            best_k = k
            best_s = 1 + s

    # Phase 2: Larger segments
    # Try segments of size 100000 at various starts
    seg_size = 100000
    # Search starts: 10^6, 10^7, ..., 10^12
    # And some randoms?
    starts = [10**6, 10**7, 10**8, 10**9, 10**10, 10**11]
    # Also try starts near 10^12
    starts.append(10**12 - seg_size)
    
    for base in starts:
        if time.time() - start_time > deadline - 5:
            break
        seg_start = base
        seg_end = min(base + seg_size - 1, 10**12)
        if seg_start < 1:
            continue
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s = find_longest_run(phi_vals)
            actual_s = seg_start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
        except Exception:
            pass

    # Phase 3: If time remains, try more segments
    if time.time() - start_time < deadline - 10 and best_k < 50:
        # Try more starts
        for i in range(10):
            if time.time() - start_time > deadline - 5:
                break
            base = 10**6 * (i + 2)
            if base > 10**12:
                break
            seg_start = base
            seg_end = min(base + seg_size - 1, 10**12)
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s = find_longest_run(phi_vals)
                actual_s = seg_start + s
                if k > best_k:
                    best_k = k
                    best_s = actual_s
            except Exception:
                pass

    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
