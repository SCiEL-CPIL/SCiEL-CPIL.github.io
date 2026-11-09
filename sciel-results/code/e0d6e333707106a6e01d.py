import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin for 150s limit

    # 1. Sieve primes up to 10^6 (sqrt(10^12))
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = np.array([i for i in range(2, limit + 1) if is_prime[i]], dtype=np.int64)

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
        # We only need primes p such that p^2 <= end (or rather p <= sqrt(end))
        # But for segmented sieve, we process all primes p <= sqrt(end).
        # Actually, we process primes p where p * p <= end.
        # For p > sqrt(end), p can appear at most once as a factor, and if it does,
        # it's the remaining factor.
        
        # Optimization: Only iterate primes up to sqrt(end)
        max_p = int(math.isqrt(end))
        
        for p in primes:
            if p > max_p:
                break
            
            # Find first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            
            # Create indices for multiples of p
            # We use numpy slicing for speed
            indices = np.arange(first_mult, end + 1, p, dtype=np.int64)
            if len(indices) == 0:
                continue
            
            idx_array = indices - start
            
            # Mask for positions where rems is divisible by p
            # Note: rems[idx_array] % p == 0
            mask = rems[idx_array] % p == 0
            valid_idx = idx_array[mask]
            
            if len(valid_idx) == 0:
                continue
            
            # Update phi: phi[n] -= phi[n] // p
            phis[valid_idx] -= phis[valid_idx] // p
            
            # Remove all factors of p from rems
            # We need to divide rems[valid_idx] by p repeatedly
            # Vectorized approach:
            while True:
                still_div = rems[valid_idx] % p == 0
                if not np.any(still_div):
                    break
                valid_idx = valid_idx[still_div]
                rems[valid_idx] //= p

        # Process remaining large prime factors
        # For most numbers, rems[i] will be 1 or a prime > max_p
        # If rems[i] > 1, it must be a prime (since all factors <= sqrt(end) are removed)
        # Wait, if rems[i] is composite, it would have a factor <= sqrt(rems[i]) <= sqrt(end).
        # So rems[i] is either 1 or prime.
        
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
        # This allows O(1) lookup and update
        # We use a sliding window approach
        last_seen = {}
        current_len = 0
        current_start_rel = 0
        
        for i in range(n):
            val = phi_vals[i]
            
            # If we've seen this value before, check if it's in the current window
            if val in last_seen:
                last_idx = last_seen[val]
                # If last occurrence is within current window, we have a collision
                if last_idx >= current_start_rel:
                    # Move start of window to after the last occurrence
                    current_start_rel = last_idx + 1
            
            last_seen[val] = i
            current_len = i - current_start_rel + 1
            
            if current_len > best_k:
                best_k = current_len
                best_s_rel = current_start_rel
                
            # Early termination if we hit the max possible
            if best_k >= 200:
                return best_k, best_s_rel
                
        return best_k, best_s_rel

    best_k = 1
    best_s = 1
    
    # Search strategy:
    # 1. Scan a large contiguous region near 10^6.
    # 2. If time permits, scan other regions.
    # Key insight: The density of distinct totients is high.
    # A collision happens when phi(a) = phi(b) for a < b in the window.
    # This is relatively rare for large n.
    # The current bottleneck is the phi computation.
    # Let's try to compute phi for a very large segment, e.g., 10^7 numbers.
    # This requires ~10^7 * (number of prime factors) operations.
    # With vectorization, this should be fast.
    
    # Segment size
    W = 2000000  # 2 million numbers
    
    # Start scanning from 10^6
    current_start = 1000000
    
    # We can scan multiple segments if time allows
    num_segments = 5
    
    for seg_idx in range(num_segments):
        if time.time() - start_time > deadline:
            break
            
        seg_start = current_start + seg_idx * W
        seg_end = seg_start + W - 1
        
        if seg_end > 10**12:
            seg_end = 10**12
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
                break
                
        except Exception:
            # If computation fails (e.g., memory), try smaller segment
            W_small = 100000
            for _ in range(10):
                if time.time() - start_time > deadline:
                    break
                seg_start = current_start + seg_idx * W_small
                seg_end = seg_start + W_small - 1
                if seg_end > 10**12:
                    break
                try:
                    phi_vals = compute_phi_segment(seg_start, seg_end)
                    k, s_rel = find_longest_run(phi_vals)
                    actual_s = seg_start + s_rel
                    if k > best_k:
                        best_k = k
                        best_s = actual_s
                except Exception:
                    break

    # If we haven't found a good run, try random spots in higher ranges
    if best_k < 100 and time.time() - start_time < deadline - 10:
        random.seed(42)
        
        # Try random starting points in [10^8, 10^12]
        for _ in range(20):
            if time.time() - start_time > deadline:
                break
            
            seg_start = random.randint(10**8, 10**12 - W)
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
