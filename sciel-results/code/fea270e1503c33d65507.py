"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_phi_sieve(n):
        """Compute phi(1..n) using a sieve. Returns array of length n+1."""
        phi = np.arange(n + 1, dtype=np.int64)
        for i in range(2, n + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi[j] -= phi[j] // i
        return phi
    
    def find_longest_distinct_run(phi_arr):
        """Find the longest run of consecutive indices where phi values are all distinct."""
        n = len(phi_arr) - 1  # indices 1..n
        best_len = 1
        best_start = 1
        
        # Sliding window with count map
        from collections import defaultdict
        count = defaultdict(int)
        window_start = 1
        
        for i in range(1, n + 1):
            val = int(phi_arr[i])
            count[val] += 1
            
            # Shrink window from left until all values are unique
            while count[val] > 1:
                left_val = int(phi_arr[window_start])
                count[left_val] -= 1
                window_start += 1
            
            current_len = i - window_start + 1
            if current_len > best_len:
                best_len = current_len
                best_start = window_start
        
        return best_len, best_start
    
    # Compute phi for a large range
    # Start with a moderate range
    N = 1000000
    phi_arr = compute_phi_sieve(N)
    elapsed = time.time() - start_time
    print(f"Computed phi up to {N} in {elapsed:.2f}s")
    
    best_len, best_start = find_longest_distinct_run(phi_arr)
    print(f"Best run in [1,{N}]: length={best_len}, start={best_start}")
    
    # If we have time, try larger ranges
    if time_remaining() > 40:
        N2 = 3000000
        phi_arr2 = compute_phi_sieve(N2)
        elapsed2 = time.time() - start_time
        print(f"Computed phi up to {N2} in {elapsed2:.2f}s")
        
        best_len2, best_start2 = find_longest_distinct_run(phi_arr2)
        print(f"Best run in [1,{N2}]: length={best_len2}, start={best_start2}")
        
        if best_len2 > best_len:
            best_len = best_len2
            best_start = best_start2
    
    if time_remaining() > 25:
        N3 = 8000000
        phi_arr3 = compute_phi_sieve(N3)
        elapsed3 = time.time() - start_time
        print(f"Computed phi up to {N3} in {elapsed3:.2f}s")
        
        best_len3, best_start3 = find_longest_distinct_run(phi_arr3)
        print(f"Best run in [1,{N3}]: length={best_len3}, start={best_start3}")
        
        if best_len3 > best_len:
            best_len = best_len3
            best_start = best_start3
    
    # Verify the result
    if best_start + best_len - 1 <= N:
        run_vals = phi_arr[best_start:best_start + best_len]
        if len(set(run_vals.tolist())) != best_len:
            best_len = 9
            best_start = 1
    
    return {'start': best_start, 'length': best_len}
# EVOLVE-BLOCK-END
