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
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def phi_segment(start, length, primes):
    """Compute phi(n) for n in [start, start+length-1] using segmented sieve."""
    phi_vals = list(range(start, start + length))
    remaining = list(range(start, start + length))
    
    max_val = start + length - 1
    sqrt_max = int(math.isqrt(max_val))
    
    for p in primes:
        if p > sqrt_max:
            break
        first = ((start + p - 1) // p) * p
        for m in range(first, start + length, p):
            idx = m - start
            phi_vals[idx] -= phi_vals[idx] // p
            while remaining[idx] % p == 0:
                remaining[idx] //= p
    
    for i in range(length):
        if remaining[i] > 1:
            phi_vals[i] -= phi_vals[i] // remaining[i]
    
    return phi_vals

def longest_distinct_run(phi_vals):
    """Find the longest subarray of consecutive distinct values in phi_vals."""
    if not phi_vals:
        return 0, 0
        
    seen = set()
    max_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0
    
    for i, v in enumerate(phi_vals):
        if v in seen:
            # Collision found, run breaks
            if cur_len > max_len:
                max_len = cur_len
                best_start = cur_start
            # Reset for new run starting at i
            seen = {v}
            cur_len = 1
            cur_start = i
        else:
            seen.add(v)
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
                best_start = cur_start
    
    # Check the last run
    if cur_len > max_len:
        max_len = cur_len
        best_start = cur_start
        
    return max_len, best_start

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    best_result = {'start': 2, 'length': 2}
    
    # Search strategy:
    # 1. Focus on high values near 10^12 where collisions are rare.
    # 2. Use large windows to amortize phi computation cost.
    # 3. Scan windows for the longest distinct run.
    
    window_size = 5000
    num_windows = 200  # Total 1,000,000 numbers searched if all are near 10^12
    
    # Define search regions
    # Region 1: Just below 10^12
    # We will step backwards from 10^12 - 199
    
    max_val_limit = 10**12 - 199
    
    # We'll search in blocks
    # Start at 10^12 - 199 - window_size
    # Then 10^12 - 199 - 2*window_size, etc.
    
    # Also, to be safe, check a few random spots in the lower ranges
    # though they are less likely to yield k=200
    
    search_starts = []
    for i in range(num_windows):
        s = max_val_limit - i * window_size
        if s < 1:
            break
        search_starts.append(s)
        
    # Add some random spots near 10^11, 10^10 just in case
    random.seed(42)
    for _ in range(50):
        s = random.randint(10**10, 10**11)
        search_starts.append(s)
    for _ in range(20):
        s = random.randint(10**8, 10**9)
        search_starts.append(s)
        
    for s in search_starts:
        if time.time() > deadline - 5:
            break
            
        if s < 1 or s + window_size - 1 > 10**12:
            continue
            
        phi_vals = phi_segment(s, window_size, primes)
        max_len, offset = longest_distinct_run(phi_vals)
        
        if max_len > best_result['length']:
            best_result = {'start': s + offset, 'length': max_len}
            
            # If we found the max possible (200), we can return immediately
            if best_result['length'] >= 200:
                # Cap at 200
                if best_result['length'] > 200:
                    best_result['length'] = 200
                    # Adjust start if necessary? 
                    # The run is s+offset to s+offset+200-1.
                    # If the found run was longer, we can just take the first 200.
                    # Or any 200.
                    pass
                return best_result

    # If we didn't find 200, try a more granular search near 10^12
    # with smaller windows but more steps, in case the large window logic missed something
    # (unlikely, but just to be safe)
    
    # Actually, if we didn't find 200 in 1M numbers near 10^12, it might not exist there?
    # Or our search was too sparse?
    # Let's try a dense search of 10,000 numbers near 10^12 with window size 1000
    # This covers 10,000 numbers.
    
    # We already covered 1,000,000 numbers.
    # Let's try searching *exactly* at 10^12 - 199 downwards with step 1
    # for the first 1000 positions, using window size 250
    # This overlaps with the large window search but ensures no gaps.
    
    for i in range(1000):
        if time.time() > deadline - 5:
            break
        s = max_val_limit - i
        if s < 1:
            break
        phi_vals = phi_segment(s, 250, primes)
        max_len, offset = longest_distinct_run(phi_vals)
        if max_len > best_result['length']:
            best_result = {'start': s + offset, 'length': max_len}
            if best_result['length'] >= 200:
                if best_result['length'] > 200:
                    best_result['length'] = 200
                return best_result

    return best_result
