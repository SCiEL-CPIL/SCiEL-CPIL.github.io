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

    def compute_phi_segment(start, length):
        """
        Compute phi for all numbers in [start, start+length) using segmented sieve.
        Returns list of phi values.
        """
        if length <= 0:
            return []
        
        # Initialize with the number itself
        phi = list(range(start, start + length))
        remaining = list(range(start, start + length))
        
        # For each prime, divide out its contribution
        for p in primes:
            if p >= start + length:
                break
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            
            for j in range(first, start + length, p):
                idx = j - start
                if remaining[idx] % p == 0:
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
                    phi[idx] -= phi[idx] // p
        
        # Handle remaining factors > sqrt(10^12)
        for i in range(length):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi

    def find_best_run_sliding(start, length, max_k=200):
        """
        Find the longest run of distinct phi values in [start, start+length)
        using a sliding window approach for O(n) instead of O(n^2).
        """
        if length <= 0:
            return 0, start
        
        # Compute phi for the range using segmented approach
        phis = compute_phi_segment(start, length)
        
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
    # 2. Use sliding window with segmented phi computation for efficiency
    # 3. Structured search: try specific regions known to have properties favoring distinct totients
    
    best_k = 1
    best_s = 1
    
    # First, check a small range to get a baseline
    if time_left() > 10:
        k, s = find_best_run_sliding(1, 5000, 200)
        if k > best_k:
            best_k, best_s = k, s

    # Structured search in promising regions
    # Longer runs of distinct totients are more likely in regions with:
    # - Numbers with diverse prime factorizations
    # - Avoiding arithmetic progressions with common factors
    
    random.seed(42)
    
    # Try multiple strategies:
    # 1. Random search in large ranges
    # 2. Search near powers of 2 (numbers with varied factorizations)
    # 3. Search in gaps between highly composite numbers
    
    while time_left() > 15:
        # Pick a random start in [10^9, 10^12]
        strategy = random.random()
        
        if strategy < 0.6:
            # Random search
            start = random.randint(10**9, 10**12 - 200)
        elif strategy < 0.8:
            # Near powers of 2
            exp = random.randint(30, 40)
            start = (1 << exp) + random.randint(-10000, 10000)
            start = max(10**9, min(10**12 - 200, start))
        else:
            # Near prime multiples (numbers just after primes have unique properties)
            base = random.randint(10**10, 10**12)
            start = base + random.randint(1, 1000)
            start = max(10**9, min(10**12 - 200, start))
        
        length = 2000  # Larger window to find longer runs
        k, s = find_best_run_sliding(start, length, 200)
        if k > best_k:
            best_k, best_s = k, s
    
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
