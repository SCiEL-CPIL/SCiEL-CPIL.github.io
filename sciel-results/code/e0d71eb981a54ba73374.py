"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # We need lpf up to N_MAX+1
    
    # 1. Sieve of Eratosthenes to find all primes up to limit
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = b'\x00' * ((limit - start) // step + 1)
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # 2. Compute Largest Prime Factor (LPF) for each number up to limit
    # Use a list for faster indexed assignment than numpy in this pattern
    lpf = [0] * (limit + 1)
    for p in primes:
        # Assign p as LPF for all multiples of p
        for m in range(p, limit + 1, p):
            lpf[m] = p
    
    # Check time - if we've used too much, fall back to a heuristic search
    if time.time() > deadline - 5:
        # Fallback: search near powers of 2 and smooth numbers
        best_n = N_MIN
        best_val = float('inf')
        # Check candidates near common smooth structures
        candidates = set()
        for k in range(13, 20):
            base = 2 ** k
            for delta in range(-100, 101):
                c = base + delta
                if N_MIN <= c <= N_MAX:
                    candidates.add(c)
        # Also check multiples of small primes
        for small_p in [2, 3, 5, 7, 11, 13]:
            for m in range(N_MIN // small_p, N_MAX // small_p + 1):
                c = m * small_p
                if N_MIN <= c <= N_MAX:
                    candidates.add(c)
        
        for n in sorted(candidates):
            if time.time() > deadline - 2:
                break
            # Compute LPF of n and n+1 by trial division
            def lpf_trial(x):
                if x <= 1:
                    return x
                result = 1
                f = 2
                while f * f <= x:
                    while x % f == 0:
                        result = f
                        x //= f
                    f += 1
                if x > 1:
                    result = max(result, x)
                return result
            
            v1 = lpf_trial(n)
            v2 = lpf_trial(n + 1)
            combined = max(v1, v2)
            if combined < best_val:
                best_val = combined
                best_n = n
        
        return best_n
    
    # 3. For each n in [N_MIN, N_MAX], compute max(lpf[n], lpf[n+1])
    #    and find the n that minimizes this value.
    # Use NumPy for vectorized computation
    lpf_np = np.array(lpf, dtype=np.int32)
    n_range = np.arange(N_MIN, N_MAX + 1, dtype=np.int32)
    v1 = lpf_np[n_range]
    v2 = lpf_np[n_range + 1]
    combined = np.maximum(v1, v2)
    
    min_idx = np.argmin(combined)
    best_n = int(n_range[min_idx])
    
    return best_n
# EVOLVE-BLOCK-END
