import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0 # Safety margin for 150s kill
    MAX_S = 10**12
    MAX_K = 200
    
    # 1. Generate primes up to 10^6 for fast phi calculation
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Efficient phi calculation
    def phi(x):
        if x == 1:
            return 1
        result, m = x, x
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    # 3. Search Strategy: Random Sampling
    # Collisions in phi(n) are rare for large n.
    # A window of 200 consecutive integers near 10^12 is very likely to have distinct totients.
    # We try random starts to find such a window quickly.
    
    best_k = 1
    best_s = 1
    
    # Try a few random windows
    # We need s + 199 <= 10^12
    max_start = MAX_S - MAX_K + 1
    
    # Use fixed seed for reproducibility, try more windows
    rng = random.Random(42)
    for _ in range(300):
        if time.time() - start_time > LIMIT:
            break
            
        s = rng.randint(1, max_start)
        
        # Check window [s, s + MAX_K - 1]
        seen = set()
        valid = True
        for i in range(s, s + MAX_K):
            p_val = phi(i)
            if p_val in seen:
                valid = False
                break
            seen.add(p_val)
            
        if valid:
            return {'start': s, 'length': MAX_K}
            
    # Fallback: deterministic scan from high end
    current_end = MAX_S
    block_size = 5000
    
    while current_end > 1:
        if time.time() - start_time > LIMIT - 5:
            break
            
        block_start = max(1, current_end - block_size + 1)
        
        phis = []
        for i in range(block_start, current_end + 1):
            phis.append(phi(i))
            
        # Find longest run of distinct values
        seen_vals = {}
        l = 0
        max_len = 0
        max_start_idx = 0
        r = 0
        for r in range(len(phis)):
            if phis[r] in seen_vals and seen_vals[phis[r]] >= l:
                l = seen_vals[phis[r]] + 1
            else:
                if r - l + 1 > max_len:
                    max_len = r - l + 1
                    max_start_idx = l
            seen_vals[phis[r]] = r
            
        if max_len >= MAX_K:
            return {'start': block_start + max_start_idx, 'length': MAX_K}
        if max_len > best_k:
            best_k = max_len
            best_s = block_start + max_start_idx
                
        current_end = block_start - 1
        
    return {'start': best_s, 'length': best_k}
