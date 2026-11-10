import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Sieve primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_segment(L, R):
        size = R - L + 1
        phi_vals = [i for i in range(L, R + 1)] # n
        res = [i for i in range(L, R + 1)]      # phi(n)
        
        for p in primes:
            if p * p > R:
                break
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            for m in range(start, R + 1, p):
                idx = m - L
                res[idx] -= res[idx] // p
                while phi_vals[idx] % p == 0:
                    phi_vals[idx] //= p
        
        for i in range(size):
            if phi_vals[i] > 1:
                res[i] -= res[i] // phi_vals[i]
                
        return res

    def find_max_run(vals):
        max_k = 0
        best_start = 0
        current_start = 0
        seen = set()
        
        for i, v in enumerate(vals):
            if v in seen:
                # Restart window
                # We need to remove elements from seen until the duplicate is removed
                # But we need to track the start index of the current window
                # A simpler way: sliding window with a count dict or just reset if window is small?
                # Since we look for max run, and max is 200, we can just check windows of size 200?
                # No, we want the MAX run.
                # If we hit a duplicate, the current run ends at i-1.
                # We can restart from i-1? No, that's not correct.
                # Example: 1, 2, 1. Run 1: 1,2 (len 2). Next is 1. Duplicate.
                # Run ending at 2 has len 2.
                # We can just keep a set and a start pointer.
                # When duplicate found at i, we shrink from left until the value at i is unique in the window.
                # But we want the max run so far.
                pass
        
        # Efficient sliding window for distinct elements
        max_len = 0
        start = 0
        counts = {}
        for i in range(len(vals)):
            v = vals[i]
            counts[v] = counts.get(v, 0) + 1
            while counts[v] > 1:
                left_v = vals[start]
                counts[left_v] -= 1
                start += 1
            current_len = i - start + 1
            if current_len > max_len:
                max_len = current_len
                best_start = start
            if max_len >= 200:
                return 200, best_start
        
        return max_len, best_start

    best_k = 0
    best_s = 1
    
    segments = [
        (10**11, 10**11 + 10**6),
        (10**12 - 10**6, 10**12),
        (10**9, 10**9 + 10**6),
        (10**8, 10**8 + 10**6)
    ]
    
    for L, R in segments:
        if time.time() > deadline:
            break
        if R > 10**12:
            R = 10**12
        if L < 1:
            L = 1
        if L >= R:
            continue
            
        phis = compute_phi_segment(L, R)
        k, s_offset = find_max_run(phis)
        s = L + s_offset
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}
