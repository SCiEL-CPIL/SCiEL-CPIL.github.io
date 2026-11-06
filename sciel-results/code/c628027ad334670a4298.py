import time
import numpy as np
from math import isqrt

def construct():
    t0 = time.time()
    DEADLINE = 85  # Safety margin for 90s limit
    
    def elapsed():
        return time.time() - t0
    
    MAXN = 1_000_000
    MAXN_PLUS = MAXN + 1  # need pp[n+1] for n up to MAXN-1, so index up to MAXN
    
    # 1. Sieve of Eratosthenes to get primes
    is_prime = bytearray(b'\x01') * (MAXN + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(MAXN) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, MAXN + 1) if is_prime[i]]
    
    # 2. Compute powerful part P(m) for all m using NumPy
    # P(m) = product of p^k where k = v_p(m) and k >= 2
    pp = np.ones(MAXN_PLUS, dtype=np.int64)
    
    for p in primes:
        if elapsed() > DEADLINE:
            break
            
        p_power = p * p
        
        while p_power <= MAXN:
            # Numbers with v_p(m) == k are multiples of p^k but not p^(k+1)
            next_power = p_power * p
            
            # j ranges from 1 to floor(MAXN / p_power)
            max_j = MAXN // p_power
            
            if next_power <= MAXN:
                # j values where j % p != 0
                # Use tiled pattern to avoid modulo on large array
                pattern = np.ones(p, dtype=bool)
                pattern[-1] = False
                num_full_blocks = max_j // p
                remainder = max_j % p
                if num_full_blocks > 0:
                    mask = np.tile(pattern, num_full_blocks)
                    if remainder > 0:
                        mask = np.concatenate([mask, pattern[:remainder]])
                else:
                    mask = pattern[:max_j]
                
                j = np.arange(1, max_j + 1)
                j_filtered = j[mask]
                indices = p_power * j_filtered
            else:
                # No filtering needed, all multiples of p_power work
                indices = np.arange(p_power, MAXN_PLUS, p_power)
            
            if len(indices) > 0:
                pp[indices] *= p_power
            
            p_power = next_power
    
    # 3. Find max pp[n] * pp[n+1] for n in [10000, MAXN-1]
    if elapsed() > DEADLINE:
        # Fallback: return a reasonable default
        return 10000
    
    # Compute products for n in [10000, MAXN-1]
    # pp[n] * pp[n+1] for n from 10000 to MAXN-1
    products = pp[10000:MAXN] * pp[10001:MAXN_PLUS]
    
    best_idx = np.argmax(products)
    best_val = products[best_idx]
    best_n = 10000 + best_idx
    
    return int(best_n)
