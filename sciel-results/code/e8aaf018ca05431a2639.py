import time
import math
from collections import defaultdict
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Use a large N to maximize collisions
    # 20 million is a good balance of time and memory
    N = 20_000_000
    
    # Fallback result if sieve fails or is too slow
    # M=336 has k=12, 14. (2 solutions)
    # M=1 has k=1. (1 solution)
    # We know seed found 3 or 4.
    # Let's try to find a better static fallback if possible?
    # For now, 336 is safe.
    fallback = {'M': 336, 'k': [12, 14]}
    
    try:
        # Allocate arrays
        # 'Q' is unsigned 64-bit
        # 'I' is unsigned 32-bit
        sigma = array('Q', [0]) * (N + 1)
        spp = array('Q', [0]) * (N + 1)
        p_power = array('Q', [0]) * (N + 1)
        lp = array('I', [0]) * (N + 1)
        
        sigma[1] = 1
        spp[1] = 1
        p_power[1] = 1
        
        primes = []
        
        # Linear sieve
        for i in range(2, N + 1):
            if lp[i] == 0:
                lp[i] = i
                primes.append(i)
                sigma[i] = 1 + i
                spp[i] = 1 + i
                p_power[i] = i
                
            for p in primes:
                if p > lp[i] or i * p > N:
                    break
                ip = i * p
                lp[ip] = p
                if i % p == 0:
                    p_power[ip] = p_power[i] * p
                    spp[ip] = spp[i] + p_power[ip]
                    sigma[ip] = sigma[i] // spp[i] * spp[ip]
                    break
                else:
                    p_power[ip] = p
                    spp[ip] = 1 + p
                    sigma[ip] = sigma[i] * (1 + p)
                    
        if time.time() > deadline - 20:
            return fallback
            
        freq = defaultdict(int)
        max_m = 10**18
        
        for k in range(1, N + 1):
            m_val = k * sigma[k]
            if m_val <= max_m:
                freq[m_val] += 1
                
        if not freq:
            return fallback
            
        best_count = 0
        best_m = 1
        for m, c in freq.items():
            if c > best_count:
                best_count = c
                best_m = m
                
        # Collect k's for best_m
        best_ks = []
        for k in range(1, N + 1):
            if k * sigma[k] == best_m:
                best_ks.append(k)
                
        if not best_ks:
            return fallback
            
        # Validate k <= sqrt(M)
        sqrt_m = math.isqrt(best_m)
        valid_ks = [k for k in best_ks if k <= sqrt_m]
        
        if not valid_ks:
            return fallback
            
        return {'M': int(best_m), 'k': [int(k) for k in valid_ks]}
        
    except Exception:
        return fallback

if __name__ == "__main__":
    result = construct()
    print(result)
