import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin before 150s kill
    
    # Generate primes up to 10^6 to handle ranges up to 10^12
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    best_k = 1
    best_s = 1
    
    def process_segment(L, R):
        nonlocal best_k, best_s
        if R > 10**12:
            R = 10**12
        if L > R:
            return
        n = R - L + 1
        if n <= 0:
            return
        # Use int64 for phi and temp
        phi = np.arange(L, R+1, dtype=np.int64)
        temp = np.arange(L, R+1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            # Find indices where temp is divisible by p
            mask = temp % p == 0
            if np.any(mask):
                phi[mask] = phi[mask] // p * (p - 1)
                temp[mask] = temp[mask] // p
        
        # Handle remaining prime factors
        mask = temp > 1
        if np.any(mask):
            q = temp[mask]
            phi[mask] = phi[mask] // q * (q - 1)
        
        # Find max run of distinct phi values using "previous occurrence" method
        phi_list = phi.tolist()
        
        # prev[i] = index of previous occurrence of phi_list[i], or -1 if none
        prev = np.full(n, -1, dtype=np.int64)
        seen = {}
        for i in range(n):
            val = phi_list[i]
            if val in seen:
                prev[i] = seen[val]
            seen[val] = i
        
        # Length of longest distinct run ending at position i:
        # If prev[i] >= 0: length = i - prev[i]
        # If prev[i] == -1: length = i + 1
        idx = np.arange(n, dtype=np.int64)
        lengths = np.where(prev >= 0, idx - prev, idx + 1)
        max_len = int(np.max(lengths))
        max_end = int(np.argmax(lengths))
        max_start = max_end - max_len + 1
        
        if max_len > best_k:
            best_k = max_len
            best_s = L + max_start
            
        return best_k, best_s

    # Search strategy: process multiple segments to maximize coverage
    # Start with smaller segments that can be processed quickly, then expand
    
    # Process [1, 10^6] first - fast
    if time.time() < deadline - 10:
        process_segment(1, 10**6)
    
    # Process [10^6+1, 2*10^6]
    if time.time() < deadline - 8:
        process_segment(10**6 + 1, 2 * 10**6)
    
    # Process [2*10^6+1, 3*10^6]
    if time.time() < deadline - 6:
        process_segment(2 * 10**6 + 1, 3 * 10**6)
    
    # Process [3*10^6+1, 4*10^6]
    if time.time() < deadline - 4:
        process_segment(3 * 10**6 + 1, 4 * 10**6)
    
    # Process [4*10^6+1, 5*10^6]
    if time.time() < deadline - 2:
        process_segment(4 * 10**6 + 1, 5 * 10**6)
    
    # If time permits, try larger segments
    if time.time() < deadline - 1:
        process_segment(5 * 10**6 + 1, 10 * 10**6)
    
    # Try a segment near 10^12 if primes allow (we have primes up to 10^6, sqrt(10^12)=10^6)
    if time.time() < deadline - 0.5:
        process_segment(10**12 - 10**6, 10**12)

    return {'start': best_s, 'length': best_k}
