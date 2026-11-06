import time
import math
import random
import sys

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(is_prime[start:limit+1:step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def compute_phi_segment(start, length, primes):
    """
    Compute exact phi(n) for n in [start, start+length-1].
    Returns list of phi values.
    """
    phi_val = list(range(start, start + length))
    rem = list(range(start, start + length))
    
    max_val = start + length - 1
    
    for p in primes:
        if p * p > max_val:
            break
        
        # Find first multiple of p in [start, start+length-1]
        r = start % p
        if r == 0:
            first = start
        else:
            first = start + (p - r)
        
        if first > max_val:
            continue
        
        for multiple in range(first, max_val + 1, p):
            idx = multiple - start
            # Only update if p is a factor of the current remainder
            # This ensures we apply the (1 - 1/p) factor exactly once for each prime factor
            if rem[idx] % p == 0:
                phi_val[idx] = phi_val[idx] // p * (p - 1)
                while rem[idx] % p == 0:
                    rem[idx] //= p

    for i in range(length):
        if rem[i] > 1:
            phi_val[i] = phi_val[i] // rem[i] * (rem[i] - 1)
            
    return phi_val

def find_longest_distinct_run(phis):
    """
    Find the longest contiguous subarray of phis with all distinct values.
    Returns (start_index, length).
    """
    n = len(phis)
    if n == 0:
        return 0, 0
    best_len = 1
    best_start = 0
    left = 0
    seen = {}
    for right in range(n):
        if phis[right] in seen and seen[phis[right]] >= left:
            left = seen[phis[right]] + 1
        seen[phis[right]] = right
        curr_len = right - left + 1
        if curr_len > best_len:
            best_len = curr_len
            best_start = left
    return best_start, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Generate primes up to 10^6 (sufficient for n up to 10^12)
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    best_k = 1
    best_start = 1
    
    random.seed(42)
    
    # Segment size for scanning
    # Larger segment = better chance of finding long run, but slower per segment
    # 10^5 is a good balance
    segment_size = 100000
    
    # Strategy: Scan random segments of size 10^5
    # We can fit ~1000 segments in 100 seconds if each takes 0.1s
    
    while time.time() < deadline - 5:
        # Pick a random start
        # Ensure we don't go out of bounds
        max_start = max_n - segment_size
        if max_start < 1:
            break
        s = random.randint(1, max_start)
        
        phis = compute_phi_segment(s, segment_size, primes)
        
        start_idx, run_len = find_longest_distinct_run(phis)
        
        if run_len > best_k:
            best_k = run_len
            best_start = s + start_idx
            
            # If we found a very long run, try to verify/extend it locally
            # But the sliding window already finds the max in the segment.
            # We might miss a run that spans two segments.
            # To handle this, we can overlap segments or check boundaries.
            # For now, random segments are likely to catch long runs.
            
            # If we hit the max possible k, we can stop early if we are confident
            if best_k == max_k:
                return {'start': best_start, 'length': best_k}
    
    return {'start': best_start, 'length': best_k}
