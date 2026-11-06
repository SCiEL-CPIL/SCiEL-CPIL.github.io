import numpy as np
import time
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
        
    def compute_phi_sieve_numpy(limit):
        """Compute phi for all numbers up to limit using NumPy."""
        phi = np.arange(limit + 1, dtype=np.int64)
        # Optimization: iterate only up to sqrt(limit) for marking? No, we need all primes.
        # But we can skip composites.
        # The current loop checks phi[i] == i, which is a prime check.
        # This is efficient enough.
        for i in range(2, limit + 1):
            if phi[i] == i:
                indices = np.arange(i, limit + 1, i)
                phi[indices] -= phi[indices] // i
        return phi

    def find_longest_distinct_run_sliding(phi_vals, start_offset=1):
        n = len(phi_vals)
        if n < 2:
            return 1, 0
            
        last_seen = {}
        left = 0
        max_len = 0
        max_start = 0
        
        for right in range(n):
            val = phi_vals[right]
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            last_seen[val] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
                
        return max_len, max_start

    best_k = 1
    best_s = 1
    
    # Try increasing sieve limits
    # 10^7 is ~1-2s for sieve + 1-2s for sliding window.
    # 5*10^7 is ~5-10s for sieve + 5-10s for sliding window.
    # 10^8 is ~10-20s for sieve + 10-20s for sliding window.
    # Total 40s. Safe.
    sieve_limits = [1000000, 10000000, 50000000, 100000000]
    
    for limit in sieve_limits:
        if time_remaining() < 30.0:
            break
            
        try:
            phi_vals = compute_phi_sieve_numpy(limit)
            k, rel_start = find_longest_distinct_run_sliding(phi_vals[1:])
            s = rel_start + 1
            
            if k > 200:
                k = 200
                # If we hit the cap, we can stop? No, maybe we found a valid 200.
                # But we want to maximize k. If we found 200, we can't do better.
                # So we can return immediately?
                # The task says "Maximize k". Max is 200.
                # If we find k=200, we are done.
                if k == 200:
                    return {'start': s, 'length': k}
                    
            if k > best_k:
                best_k = k
                best_s = s
        except MemoryError:
            break
        except Exception:
            break
            
    # Ensure valid return
    if best_k < 1:
        best_k = 1
        best_s = 1
        
    if best_k > 200:
        best_k = 200
        
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
