import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    max_n = 10**12
    best_k = 1
    best_s = 1
    
    # Segmented sieve approach: compute phi(n) for consecutive n in segments
    # Using a sliding window to track distinct totients
    
    # Pre-compute primes up to sqrt(max_n) for segmented sieve
    # We need primes up to ~10^6 for numbers up to 10^12
    SIEVE_LIMIT = int(math.isqrt(max_n)) + 1
    
    # Generate primes up to SIEVE_LIMIT
    def generate_primes(limit):
        if limit < 2:
            return []
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.array([i for i in range(2, limit + 1) if is_prime[i]])
    
    primes = generate_primes(SIEVE_LIMIT)
    
    # Segmented phi computation with sliding window
    def find_best_run_segmented(segment_size=10**6, max_segments=50):
        nonlocal best_k, best_s
        nonlocal start_time, DEADLINE
        
        for seg_idx in range(max_segments):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            seg_start = seg_idx * segment_size + 1
            seg_end = min(seg_start + segment_size - 1, max_n)
            if seg_start > max_n:
                break
            
            seg_len = seg_end - seg_start + 1
            
            # Initialize phi values for this segment
            phi_vals = np.arange(seg_start, seg_end + 1, dtype=np.int64)
            # For each prime p, multiply phi by (1 - 1/p) for multiples of p
            for p in primes:
                if p > seg_end:
                    break
                # Find first multiple of p in [seg_start, seg_end]
                first_mult = ((seg_start + p - 1) // p) * p
                # Apply phi adjustment: phi[n] *= (p-1) / p
                # To avoid floating point, do: phi[n] = phi[n] // p * (p-1)
                indices = np.arange(first_mult, seg_end + 1, p)
                phi_vals[indices - seg_start] = (phi_vals[indices - seg_start] // p) * (p - 1)
            
            # Now phi_vals[i] = phi(seg_start + i)
            # Use sliding window to find longest run of distinct values
            # Window size up to 200
            
            window_size = 200
            seen = set()
            current_run = 0
            seg_best_k = 1
            seg_best_s = seg_start
            
            for i in range(seg_len):
                val = int(phi_vals[i])
                if val in seen:
                    # Remove phi_vals[i - current_run] from window if we're at window boundary
                    if current_run >= window_size:
                        # This shouldn't happen as we break at 200
                        pass
                    current_run = 0
                    # Reset: we need to restart the window
                    # Actually, let's use a different approach: track the start of current run
                    # When we see a duplicate, the run ends at i-1
                    # But we need to be more careful about what to remove from seen
                    # Let's use a queue-based approach
                    pass
                else:
                    seen.add(val)
                    current_run += 1
                    if current_run > seg_best_k:
                        seg_best_k = current_run
                        seg_best_s = seg_start + i - current_run + 1
                    if current_run >= window_size:
                        break
            
            # The above logic is flawed. Let me redo with a proper sliding window.
            # Actually, let's just scan linearly and track the current run start.
            
            current_run_start = 0
            seen_set = set()
            for i in range(seg_len):
                val = int(phi_vals[i])
                if val in seen_set:
                    # Duplicate found, reset run
                    current_run_start = i + 1
                    seen_set = set()
                else:
                    seen_set.add(val)
                    run_len = i - current_run_start + 1
                    if run_len > seg_best_k:
                        seg_best_k = run_len
                        seg_best_s = seg_start + current_run_start
                    if run_len >= window_size:
                        break
            
            if seg_best_k > best_k:
                best_k = seg_best_k
                best_s = seg_best_s
            
            if best_k >= 200:
                break
    
    # Run segmented search
    find_best_run_segmented(segment_size=2*10**6, max_segments=20)
    
    # Also check small starts directly (quick sanity)
    for s in range(1, 100):
        if time.time() - start_time > DEADLINE - 5:
            break
        seen = set()
        k = 0
        for i in range(200):
            v = s + i
            if v > max_n:
                break
            # Compute phi(v) using primes
            n = v
            result = n
            temp = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    result -= result // p
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                result -= result // temp
            if result in seen:
                break
            seen.add(result)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
    
    return {'start': best_s, 'length': best_k}
