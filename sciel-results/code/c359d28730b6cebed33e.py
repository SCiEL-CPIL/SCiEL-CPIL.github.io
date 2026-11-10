import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 118.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Step 1: Sieve primes up to 10^6 for fast factorization
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def phi_single(n):
        """Compute phi(n) using precomputed primes."""
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
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
        using a sliding window approach for O(n) instead of O(n^2).
        """
        if length <= 0:
            return 0, start
        
        # Compute phi for the range
        phis = [phi_single(start + i) for i in range(length)]
        
        # Sliding window: maintain a window [left, right] where all phi values are distinct
        # When we find a duplicate, shrink from the left
        best_k = 1
        best_s = start
        left = 0
        freq = {}
        
        for right in range(length):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            # Shrink window from left until all values are distinct
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
            # Cap at max_k
            if best_k >= max_k:
                break
        
        return min(best_k, max_k), best_s

    # Strategy:
    # 1. Search in large ranges near 10^12 where longer runs are more likely
    # 2. Use sliding window for efficiency
    # 3. Try multiple window sizes and locations
    
    best_k = 1
    best_s = 1
    
    # First, check a small range to get a baseline
    if time_left() > 10:
        k, s = find_best_run_sliding(1, 5000, 200)
        if k > best_k:
            best_k, best_s = k, s

    # Random search in large ranges
    random.seed(42)
    num_tries = 0
    while time_left() > 15:
        # Pick a random start in [10^9, 10^12]
        start = random.randint(10**9, 10**12 - 200)
        length = 2000  # Larger window to find longer runs
        k, s = find_best_run_sliding(start, length, 200)
        if k > best_k:
            best_k, best_s = k, s
        num_tries += 1
    
    # Also try to extend the best run found
    if best_k < 200 and time_left() > 10:
        # Try extending left
        left_start = max(1, best_s - 2000)
        left_len = best_s - left_start + best_k + 2000
        k_left, s_left = find_best_run_sliding(left_start, left_len, 200)
        if k_left > best_k:
            best_k, best_s = k_left, s_left
        
        # Try extending right
        right_end = min(10**12, best_s + best_k + 2000)
        right_start = best_s
        right_len = right_end - right_start + 1
        k_right, s_right = find_best_run_sliding(right_start, right_len, 200)
        if k_right > best_k:
            best_k, best_s = k_right, s_right

    return {'start': best_s, 'length': best_k}
