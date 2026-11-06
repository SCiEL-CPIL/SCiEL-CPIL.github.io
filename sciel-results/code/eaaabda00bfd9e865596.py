"""Rewritten program: search for long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def sieve_primes(limit):
        """Sieve primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # We need primes up to sqrt(10^12) = 10^6 for the segmented approach
    primes = sieve_primes(1000000)
    
    def compute_phi_range(start, length):
        """Compute phi for consecutive integers start, start+1, ..., start+length-1.
        Uses a segmented approach: for each prime, mark its multiples in the range.
        After processing all primes <= 10^6, any remaining factor > 1 is prime."""
        if length <= 0:
            return []
        
        # Initialize: phi[i] = start + i
        phi_vals = [start + i for i in range(length)]
        
        # For each prime, update phi for multiples in range
        for p in primes:
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            if first > start + length - 1:
                break  # No more multiples in range for larger primes
            # Update phi for all multiples of p in range
            for m in range(first, start + length, p):
                idx = m - start
                phi_vals[idx] -= phi_vals[idx] // p
        
        # Handle remaining prime factors > 10^6
        # After dividing by all primes <= 10^6, if the remaining value > 1,
        # it must be a prime (since (10^6)^2 = 10^12 >= max n)
        for i in range(length):
            n = start + i
            temp = n
            # Divide out all small prime factors
            for p in primes:
                if p * p > temp:
                    break
                while temp % p == 0:
                    temp //= p
            if temp > 1:
                # temp is a prime factor > 10^6
                phi_vals[i] -= phi_vals[i] // temp
        
        return phi_vals
    
    def find_longest_distinct_run(phi_vals, offset, max_k=200):
        """Find the longest run of distinct phi values in phi_vals.
        Returns (length, start_index_in_original)."""
        best_k = 1
        best_start = offset
        n = len(phi_vals)
        
        # For each starting position, extend as far as possible
        for s in range(min(n, 5000)):  # Limit search for speed
            if time.time() > deadline - 5:
                break
            seen = set()
            k = 0
            while s + k < n and s + k < offset + max_k:
                v = phi_vals[s + k]
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_start = offset + s
                if best_k >= max_k:
                    break
        
        return best_k, best_start
    
    best_k = 1
    best_s = 1
    
    # Strategy: Focus on large ranges where phi values are well-spread
    # For n ~ 10^11 to 10^12, phi(n) is typically ~ n * (1-1/p1) * (1-1/p2) * ...
    # Consecutive numbers tend to have different prime factorizations,
    # so their phi values are likely distinct for long runs.
    
    # Phase 1: Quick baseline in small range
    if time.time() < deadline - 100:
        small_start = 1
        small_len = 20000
        phi_small = compute_phi_range(small_start, small_len)
        k, s = find_longest_distinct_run(phi_small, small_start)
        if k > best_k:
            best_k, best_s = k, s
    
    # Phase 2: Search in medium-large ranges
    # These ranges are chosen to be in the 10^9 to 10^11 range
    medium_starts = [
        10**9, 5*10**9, 10**10, 5*10**10,
    ]
    
    for start in medium_starts:
        if time.time() > deadline - 60:
            break
        if start > 10**12 - 200:
            continue
        length = 50000
        phi_med = compute_phi_range(start, length)
        k, s = find_longest_distinct_run(phi_med, start)
        if k > best_k:
            best_k, best_s = k, s
    
    # Phase 3: Focus on large ranges 10^11 to 10^12
    # This is where we expect the longest runs
    large_starts = [
        10**11, 2*10**11, 3*10**11, 5*10**11, 7*10**11,
        10**12 - 100000, 10**12 - 500000, 10**12 - 1000000,
    ]
    
    for start in large_starts:
        if time.time() > deadline - 30:
            break
        if start < 1 or start > 10**12 - 200:
            continue
        length = 100000
        phi_large = compute_phi_range(start, length)
        k, s = find_longest_distinct_run(phi_large, start)
        if k > best_k:
            best_k, best_s = k, s
    
    # Phase 4: Random sampling in the large range
    random.seed(12345)
    for _ in range(15):
        if time.time() > deadline - 10:
            break
        start = random.randint(10**11, 10**12 - 100000)
        length = 100000
        phi_rand = compute_phi_range(start, length)
        k, s = find_longest_distinct_run(phi_rand, start)
        if k > best_k:
            best_k, best_s = k, s
    
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
# EVOLVE-BLOCK-END
