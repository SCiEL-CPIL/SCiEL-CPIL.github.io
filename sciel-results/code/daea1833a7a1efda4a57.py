"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate primes up to 10^6 using sieve
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    primes_set = set(primes)
    
    # Function to compute phi(n) for n up to ~10^12
    def phi(n):
        result = n
        x = n
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                while x % p == 0:
                    x //= p
                result -= result // p
        if x > 1:
            result -= result // x
        return result
    
    # Strategy: search for long runs of consecutive integers with distinct totients
    # Key insight: phi(n) is even for n > 2, and phi(n) < n for n > 1
    # Collisions happen when phi(a) = phi(b) for a != b
    # We want to find intervals [s, s+k-1] where all phi values are distinct
    
    # Approach: scan through ranges, computing phi values and tracking distinctness
    # Use a sliding window approach with early termination
    
    best_start = 1
    best_len = 1
    
    # First, do a quick scan of small numbers to find good starting points
    # Then focus on promising regions
    
    # For small n (up to ~100000), we can compute phi quickly
    max_scan = 200000
    phi_vals = [0] * (max_scan + 1)
    phi_vals[1] = 1
    for i in range(2, max_scan + 1):
        phi_vals[i] = i
    for p in primes:
        if p > max_scan:
            break
        for multiple in range(p, max_scan + 1, p):
            phi_vals[multiple] -= phi_vals[multiple] // p
    
    # Find best run in the small range
    for s in range(1, max_scan - 200):
        seen = set()
        k = 0
        while s + k <= max_scan and phi_vals[s + k] not in seen:
            seen.add(phi_vals[s + k])
            k += 1
        if k > best_len:
            best_len = k
            best_start = s
        if time.time() > deadline:
            break
    
    # Now search in larger ranges using the phi function
    # Focus on ranges where we might find longer runs
    # Strategy: try ranges starting at various points, especially near powers of 2 or primes
    
    # Try to extend from the best found so far
    if best_len > 1:
        # Try to find even longer runs by searching nearby
        for offset in range(-1000, 1001, 100):
            s = best_start + offset
            if s < 1:
                continue
            if time.time() > deadline:
                break
            seen = set()
            k = 0
            while k < 200 and s + k <= 10**12:
                val = phi(s + k)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            if k > best_len:
                best_len = k
                best_start = s
    
    # Also try some heuristic starting points
    # Numbers just after prime powers often have interesting phi behavior
    test_starts = [2**20, 2**20 + 100, 3**12, 5**10, 7**8, 10**6, 10**6 + 1000, 
                   10**7, 10**7 + 500, 10**8, 10**8 + 1000, 10**9, 10**9 + 10000]
    
    for s in test_starts:
        if s < 1 or s + 200 > 10**12:
            continue
        if time.time() > deadline:
            break
        seen = set()
        k = 0
        while k < 200 and s + k <= 10**12:
            val = phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_len:
            best_len = k
            best_start = s
    
    # Final check: make sure our answer is valid
    if best_len > 200:
        best_len = 200
    if best_start + best_len - 1 > 10**12:
        best_len = min(best_len, int(10**12 - best_start + 1))
    
    return {'start': best_start, 'length': best_len}
# EVOLVE-BLOCK-END
