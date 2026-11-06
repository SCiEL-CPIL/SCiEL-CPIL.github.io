import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # We can try to go up to 10^8 or 10^9 if fast enough
    # 10^8 takes ~2-3 seconds for the sieve in Python
    # Let's try 10^8 first.
    MAX_N = 10**8 
    
    limit = int(math.isqrt(MAX_N))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    best_k = 1
    best_s = 1
    
    SEG_SIZE = 10**6
    
    # Buffer for boundary handling
    recent_vals = []
    recent_starts = []
    
    for L in range(1, MAX_N + 1, SEG_SIZE):
        if time.time() - start_time > DEADLINE:
            break
            
        R = min(L + SEG_SIZE - 1, MAX_N)
        n = R - L + 1
        
        phi = list(range(L, R + 1))
        
        for p in primes:
            if p * p > R:
                break
            start = ((L + p - 1) // p) * p
            if start < L:
                start += p
            for j in range(start, R + 1, p):
                if phi[j - L] % p == 0:
                    phi[j - L] -= phi[j - L] // p
        
        for i in range(n):
            if phi[i] == L + i:
                if L + i > 1:
                    phi[i] = L + i - 1
                else:
                    phi[i] = 1
        
        # Append to history
        for i in range(n):
            recent_vals.append(phi[i])
            recent_starts.append(L + i)
            
        # Trim to 200
        if len(recent_vals) > 200:
            excess = len(recent_vals) - 200
            recent_vals = recent_vals[excess:]
            recent_starts = recent_starts[excess:]
            
        # Scan recent_vals for longest run of distinct values
        seen = set()
        max_run_in_window = 0
        max_run_start_in_window = 0
        
        cur_run_len = 0
        cur_run_start_idx = 0
        
        for i in range(len(recent_vals)):
            v = recent_vals[i]
            if v in seen:
                if cur_run_len > max_run_in_window:
                    max_run_in_window = cur_run_len
                    max_run_start_in_window = cur_run_start_idx
                seen = set()
                cur_run_len = 0
                cur_run_start_idx = i
            seen.add(v)
            cur_run_len += 1
            
        if cur_run_len > max_run_in_window:
            max_run_in_window = cur_run_len
            max_run_start_in_window = cur_run_start_idx
            
        if max_run_in_window > best_k:
            best_k = max_run_in_window
            best_s = recent_starts[max_run_start_in_window]

    return {'start': best_s, 'length': best_k}
