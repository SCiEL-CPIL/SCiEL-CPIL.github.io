import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 82s safety margin (90s kill limit)
    
    N = 1000000
    limit = N + 1
    
    lpf = np.zeros(limit + 1, dtype=np.int64)
    
    # NumPy sieve for Largest Prime Factor
    sieve_complete = True
    for i in range(2, limit + 1):
        if time.time() > deadline:
            sieve_complete = False
            break
        if lpf[i] == 0:
            lpf[i::i] = i
    
    if not sieve_complete:
        # Fallback: linear scan with partial sieve
        best_val = float('inf')
        best_n = 10000
        for n in range(10000, N + 1):
            a = int(lpf[n])
            b = int(lpf[n+1])
            if a == 0 or b == 0:
                continue
            val = a if a > b else b
            if val < best_val:
                best_val = val
                best_n = n
        return best_n
    
    # Vectorized scan for minimum max prime factor
    vals = np.maximum(lpf[10000:N+1], lpf[10001:N+2])
    min_idx = int(np.argmin(vals))
    best_n = 10000 + min_idx
    best_val = int(vals[min_idx])
    
    # Verify with exact trial division and try to find better candidates
    # near the found n by checking neighbors with smaller max prime factors
    
    def largest_prime_factor(n):
        """Compute largest prime factor via trial division."""
        if n <= 1:
            return 1
        orig = n
        lp = 1
        # Check factor 2
        while n % 2 == 0:
            lp = 2
            n //= 2
        # Check odd factors up to sqrt(n)
        f = 3
        while f * f <= n:
            while n % f == 0:
                lp = f
                n //= f
            f += 2
        if n > 1:
            lp = n
        return lp
    
    # Try to find a better n by scanning a window around best_n
    # and also checking if any n has max(lpf[n], lpf[n+1]) < best_val
    # The sieve should be accurate, but let's verify and potentially refine
    
    # Check a wider window for potential improvements
    window = 1000
    lo = max(10000, best_n - window)
    hi = min(N, best_n + window)
    
    for n in range(lo, hi + 1):
        if time.time() > deadline:
            break
        a = largest_prime_factor(n)
        b = largest_prime_factor(n + 1)
        val = a if a > b else b
        if val < best_val:
            best_val = val
            best_n = n
    
    # If we still have time, do a broader scan
    if time.time() < deadline - 5.0:
        # Scan in chunks, checking only candidates that could beat best_val
        # based on the sieve (sieve values are exact for the largest prime factor)
        # So we can use the sieve to filter: only check n where max(lpf[n], lpf[n+1]) <= best_val
        candidates = np.where(vals <= best_val)[0]
        # Check these candidates with exact trial division to verify
        for idx in candidates:
            if time.time() > deadline:
                break
            n = 10000 + int(idx)
            a = largest_prime_factor(n)
            b = largest_prime_factor(n + 1)
            val = a if a > b else b
            if val < best_val:
                best_val = val
                best_n = n
    
    return best_n
