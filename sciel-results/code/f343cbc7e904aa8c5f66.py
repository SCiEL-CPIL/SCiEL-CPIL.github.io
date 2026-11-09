import numpy as np
import time

def construct():
    start_time = time.time()
    MAX_N = 10**12
    MAX_K = 200
    
    # Generate primes up to 10^6 (sqrt of 10^12)
    def get_primes(limit):
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.where(sieve)[0]
    
    primes = get_primes(10**6)
    
    # Segment size - larger is better for efficiency, but needs to fit in memory
    SEG_SIZE = 10**7
    
    best_k = 1
    best_s = 1
    
    L = 1
    # Keep track of the last T phi values and their corresponding numbers
    T = MAX_K - 1  # We need to check runs that span segment boundaries
    prev_phis = []  # List of phi values for the last T numbers
    prev_nums = []  # List of numbers corresponding to prev_phis
    
    while L <= MAX_N:
        if time.time() - start_time > 110:
            break
            
        R = min(L + SEG_SIZE - 1, MAX_N)
        N = R - L + 1
        
        # Initialize phi array with the numbers themselves
        phi_arr = np.arange(L, R + 1, dtype=np.int64)
        
        # Apply primes to compute phi
        for p in primes:
            if p > R:
                break
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            # Indices in phi_arr
            idx = np.arange(start, R + 1, p) - L
            # Update phi: phi[x] = phi[x] / p * (p-1)
            phi_arr[idx] = (phi_arr[idx] // p) * (p - 1)
        
        # Handle remaining large prime factors
        # After dividing by all small primes, if phi_arr[i] > 1, it's a prime factor
        mask_large = phi_arr > 1
        if np.any(mask_large):
            phi_arr[mask_large] = (phi_arr[mask_large] // phi_arr[mask_large]) * (phi_arr[mask_large] - 1)
        
        curr_phis = phi_arr.tolist()
        curr_nums = list(range(L, R + 1))
        
        # Combine with previous tail
        full_phis = prev_phis + curr_phis
        full_nums = prev_nums + curr_nums
        
        # Sliding window for distinct values
        # We want the longest subarray with distinct elements
        seen = {}
        left = 0
        local_max_k = 0
        local_max_start = 0
        
        for right, val in enumerate(full_phis):
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = right
            curr_k = right - left + 1
            if curr_k > local_max_k:
                local_max_k = curr_k
                local_max_start = left
        
        # Determine actual start number
        if local_max_k > best_k:
            actual_start = full_nums[local_max_start]
            
            # Verify constraints
            if actual_start >= 1 and actual_start + local_max_k - 1 <= MAX_N and local_max_k <= MAX_K:
                best_k = local_max_k
                best_s = actual_start
        
        # Update tail - keep the last T values
        if len(full_phis) >= T:
            prev_phis = full_phis[-T:]
            prev_nums = full_nums[-T:]
        else:
            prev_phis = full_phis
            prev_nums = full_nums
        
        L = R + 1
        
    return {'start': best_s, 'length': best_k}
