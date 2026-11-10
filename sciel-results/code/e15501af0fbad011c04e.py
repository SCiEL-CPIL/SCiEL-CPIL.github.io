import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Step 1: Sieve primes up to 10^6 for fast factorization
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = np.array([i for i in range(2, limit + 1) if is_prime[i]], dtype=np.int64)
    primes_list = primes.tolist()

    def compute_phi_segment(start, length):
        """
        Compute phi(x) for x in [start, start+length-1] using segmented sieve.
        Returns a list of phi values.
        """
        if length <= 0:
            return []
        
        # Initialize with the numbers themselves
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        
        # For each prime, apply the factor (1 - 1/p) to all multiples
        for p in primes_list:
            p_int = int(p)
            # Find first multiple of p in [start, start+length)
            first = ((start + p_int - 1) // p_int) * p_int
            if first < start:
                first += p_int
            
            if first >= start + length:
                continue
            
            # Apply phi update: val = val // p * (p-1)
            # Vectorized over the indices
            indices = np.arange(first - start, length, p_int)
            phi_vals[indices] = phi_vals[indices] // p_int * (p_int - 1)
        
        # Handle remaining large prime factors
        # After processing all primes <= 10^6, if the original number had a prime factor > 10^6,
        # we need to apply that factor too.
        # The original number is start + i. After dividing out all small prime factors,
        # the remainder should be 1 or a prime > 10^6.
        # But we've been modifying phi_vals, not tracking the original number separately.
        # 
        # Alternative: track the "remaining" part of each number separately.
        # Let's redo: keep a copy of the original numbers to track remainders.
        
        # Actually, let's use a cleaner approach:
        # rem[i] tracks the remaining part of (start+i) after removing small prime factors
        # phi[i] tracks the running phi value
        
        # Restart with proper tracking
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        rem = np.arange(start, start + length, dtype=np.int64)
        
        for p in primes_list:
            p_int = int(p)
            first = ((start + p_int - 1) // p_int) * p_int
            if first < start:
                first += p_int
            if first >= start + length:
                continue
            
            indices = np.arange(first - start, length, p_int)
            # Apply phi factor
            phi_vals[indices] = phi_vals[indices] // p_int * (p_int - 1)
            # Remove all factors of p from rem
            mask = indices.copy()
            while len(mask) > 0:
                still_divisible = rem[mask] % p_int == 0
                if not np.any(still_divisible):
                    break
                mask = mask[still_divisible]
                rem[mask] = rem[mask] // p_int
        
        # For any remaining rem[i] > 1, it's a prime factor > 10^6
        # Apply the factor (1 - 1/rem[i])
        large_mask = rem > 1
        if np.any(large_mask):
            phi_vals[large_mask] = phi_vals[large_mask] // rem[large_mask] * (rem[large_mask] - 1)
        
        return phi_vals.tolist()

    def phi_single(n):
        """Compute phi(n) using precomputed primes for single values."""
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes_list:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    def find_best_run_sliding(start, length, max_k=200):
        """
        Find the longest run of distinct phi values in [start, start+length)
        using a sliding window approach.
        """
        if length <= 0:
            return 0, start
        
        phis = compute_phi_segment(start, length)
        
        # Sliding window
        best_k = 1
        best_s = start
        left = 0
        freq = {}
        
        for right in range(length):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            while freq[val] > 1:
                left_val = phis[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = start + left
            if best_k >= max_k:
                break
        
        return min(best_k, max_k), best_s

    best_k = 1
    best_s = 1
    
    # Phase 1: Quick baseline search in small range
    if time_left() > 20:
        k, s = find_best_run_sliding(1, 10000, 200)
        if k > best_k:
            best_k, best_s = k, s

    # Phase 2: Random search in large ranges with larger windows
    random.seed(42)
    num_tries = 0
    while time_left() > 20:
        # Pick random start in [10^8, 10^12 - 5000]
        start = random.randint(10**8, 10**12 - 5000)
        # Use a window of 5000 to catch longer runs
        length = 5000
        k, s = find_best_run_sliding(start, length, 200)
        if k > best_k:
            best_k, best_s = k, s
        num_tries += 1
    
    # Phase 3: Local extension around best found
    if best_k < 200 and time_left() > 15:
        # Try extending with a larger window around the best
        margin = 5000
        left_start = max(1, best_s - margin)
        right_end = min(10**12, best_s + best_k + margin)
        full_start = left_start
        full_len = right_end - left_start + 1
        k, s = find_best_run_sliding(full_start, full_len, 200)
        if k > best_k:
            best_k, best_s = k, s

    # Phase 4: If still short, try a few more targeted searches near the best
    if best_k < 200 and time_left() > 10:
        for _ in range(5):
            if time_left() < 5:
                break
            offset = random.randint(-10000, 10000)
            search_start = max(1, best_s + offset - 2500)
            k, s = find_best_run_sliding(search_start, 5000, 200)
            if k > best_k:
                best_k, best_s = k, s

    return {'start': best_s, 'length': best_k}
