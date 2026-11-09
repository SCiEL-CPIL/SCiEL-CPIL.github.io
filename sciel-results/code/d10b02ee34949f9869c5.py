"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def compute_primes_up_to(n):
        """Compute all primes up to n using simple sieve."""
        if n < 2:
            return []
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    def compute_phi_segment(start, end, primes):
        """
        Compute phi for all numbers from start to end using segmented sieve.
        Returns a list of phi values for [start, end].
        """
        length = end - start + 1
        # Initialize: phi[i] = i - start + 1 (the number itself)
        phi = list(range(start, end + 1))
        # Also track remaining factor for each number
        rem = list(range(start, end + 1))
        
        for p in primes:
            if p * p > end:
                break
            # Find the first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first < p:
                first = p
            for j in range(first, end + 1, p):
                idx = j - start
                if rem[idx] % p == 0:
                    phi[idx] = phi[idx] // p * (p - 1)
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        # Handle remaining prime factors > sqrt(end)
        for i in range(length):
            if rem[i] > 1:
                phi[i] = phi[i] // rem[i] * (rem[i] - 1)
        
        return phi
    
    def scan_for_best_run(phi_vals, offset, max_k=200):
        """
        Scan phi_vals for the longest run of distinct values.
        phi_vals[i] corresponds to number offset + i.
        Uses a dictionary to track last seen position for O(N) scanning.
        Returns (best_k, best_start_number)
        """
        best_k = 1
        best_s = offset
        n = len(phi_vals)
        
        # last_seen[v] = index of last occurrence of value v
        # Use sliding window with two pointers
        last_seen = {}
        left = 0
        for right in range(n):
            v = phi_vals[right]
            if v in last_seen:
                # Move left pointer past the previous occurrence
                if last_seen[v] >= left:
                    left = last_seen[v] + 1
            last_seen[v] = right
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = offset + left
                if best_k >= max_k:
                    break
        
        return best_k, best_s
    
    best_k = 1
    best_s = 1
    
    # Precompute primes up to a reasonable limit
    # We need primes up to sqrt(max_range) for the segmented sieve
    max_primes_limit = 1000000  # sqrt(10^12) = 10^6
    primes = compute_primes_up_to(max_primes_limit)
    
    # Strategy: try different ranges, from small to large
    # Focus on ranges where long runs might exist
    # Known: long runs of distinct totients are rare and tend to be at larger numbers
    
    # Try several range sizes and starting positions
    # First, try from 1 upward with increasing range sizes
    range_configs = [
        (1, 500000),
        (1, 2000000),
        (1, 5000000),
        (1, 10000000),
        (100000, 5000000),
        (500000, 10000000),
        (1000000, 20000000),
        (5000000, 50000000),
    ]
    
    for start_pos, end_pos in range_configs:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        # Check if this range is feasible given remaining time
        range_size = end_pos - start_pos + 1
        # Estimate time: roughly proportional to range_size / sqrt(end_pos) * log(log(end_pos))
        # Be conservative
        if range_size > 10000000 and elapsed > deadline - 60:
            break
        if range_size > 5000000 and elapsed > deadline - 40:
            break
        if range_size > 2000000 and elapsed > deadline - 25:
            break
        
        # Compute phi for this segment
        phi_vals = compute_phi_segment(start_pos, end_pos, primes)
        
        # Scan for longest run
        k_found, s_found = scan_for_best_run(phi_vals, start_pos, max_k=200)
        
        if k_found > best_k:
            best_k = k_found
            best_s = s_found
        
        if best_k >= 200:
            break
    
    # If we still have time, try some random large ranges
    elapsed = time.time() - start_time
    if elapsed < deadline - 30 and best_k < 200:
        import random
        random.seed(42)
        # Try a few random ranges in the high range
        for _ in range(5):
            elapsed = time.time() - start_time
            if elapsed > deadline - 20:
                break
            # Random start in [10^6, 10^8]
            r_start = random.randint(1000000, 100000000)
            r_end = r_start + random.randint(1000000, 10000000)
            if r_end > 10**12:
                continue
            phi_vals = compute_phi_segment(r_start, r_end, primes)
            k_found, s_found = scan_for_best_run(phi_vals, r_start, max_k=200)
            if k_found > best_k:
                best_k = k_found
                best_s = s_found
            if best_k >= 200:
                break
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
