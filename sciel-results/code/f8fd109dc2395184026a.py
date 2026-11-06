import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    # 1. Sieve primes up to 10^6
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    max_n = 10**12
    best_k = 1
    best_s = 1
    
    # Segment size
    # 10^6 is safe for memory and speed balance
    SEG_SIZE = 5_000_000
    
    # We will scan segments. 
    # To maximize chance, we can scan from 1 upwards, or randomly.
    # Random scanning might hit "good" spots faster if they are sparse.
    # However, distinct totient runs are likely rare and short.
    # Scanning sequentially from 1 covers the most "ground" deterministically.
    # Let's try a mix: first few segments from start, then random.
    
    def process_segment(L, R):
        """
        Computes phi(n) for n in [L, R] and finds the longest run of distinct values.
        Returns (max_k, start_offset_in_segment)
        """
        N = R - L + 1
        # Initialize phi and rem
        # Using list of ints
        phi = [0] * N
        rem = [0] * N
        
        for i in range(N):
            val = L + i
            phi[i] = val
            rem[i] = val
        
        # Sieve with primes
        for p in primes:
            if p * p > R:
                break
            
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            
            # Iterate through multiples
            # Optimization: local variables
            phi_local = phi
            rem_local = rem
            p_local = p
            
            for j in range(start, R + 1, p_local):
                idx = j - L
                # Divide out p from rem
                r = rem_local[idx]
                if r % p_local == 0:
                    # Update phi: phi(n) = n * (1-1/p) ... 
                    # We are doing phi[i] -= phi[i] // p
                    # This is correct if phi[i] still has the factor p
                    phi_local[idx] -= phi_local[idx] // p_local
                    
                    # Remove p from rem
                    while r % p_local == 0:
                        r //= p_local
                    rem_local[idx] = r
        
        # Handle remaining prime factors
        for i in range(N):
            r = rem[i]
            if r > 1:
                # r is a prime factor
                phi[i] -= phi[i] // r
        
        # Find longest run of distinct phi values
        seen = set()
        max_run = 0
        cur_run = 0
        best_start_local = 0
        
        for i in range(N):
            val = phi[i]
            if val in seen:
                # Collision, reset run
                if cur_run > max_run:
                    max_run = cur_run
                    best_start_local = i - cur_run
                seen.clear()
                seen.add(val)
                cur_run = 1
            else:
                seen.add(val)
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
                    best_start_local = i - cur_run + 1
        
        # Check end of segment
        if cur_run > max_run:
            max_run = cur_run
            best_start_local = N - cur_run
            
        return max_run, best_start_local

    # Search Strategy
    # 1. Scan first few segments from 1
    # 2. Then random segments
    
    # To avoid duplicate work, we track visited segments
    # But with 10^12 range, random collisions are unlikely if we use large steps
    
    # First, scan from 1
    L = 1
    while L <= max_n:
        if time.time() - start_time > DEADLINE:
            break
            
        R = min(L + SEG_SIZE - 1, max_n)
        k, offset = process_segment(L, R)
        
        if k > best_k:
            best_k = k
            best_s = L + offset
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
        
        L = R + 1
        
        # If we've scanned enough (e.g., 10^8), switch to random
        if L > 10**8:
            break
    
    # Random search
    random.seed(42)
    while time.time() - start_time < DEADLINE:
        # Pick a random start
        s = random.randint(1, max_n - SEG_SIZE)
        R = s + SEG_SIZE - 1
        if R > max_n:
            R = max_n
            s = R - SEG_SIZE + 1
            if s < 1:
                s = 1
                R = min(SEG_SIZE, max_n)
        
        k, offset = process_segment(s, R)
        
        if k > best_k:
            best_k = k
            best_s = s + offset
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
                
    return {'start': best_s, 'length': best_k}
