import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Known good baseline
    best_M = 333312
    best_ks = [336, 372, 434]
    
    # Search for better M by iterating k
    # Group by M = k * sigma(k)
    m_to_ks = {}
    
    # Use a linear sieve for SPF - much faster than trial division sieve
    def compute_spf(n):
        spf = list(range(n + 1))
        primes = []
        for i in range(2, n + 1):
            if spf[i] == i:
                primes.append(i)
            for p in primes:
                if p > spf[i] or i * p > n:
                    break
                spf[i * p] = p
        return spf, primes
    
    # Try progressively larger max_k values
    prev_max_k = 0
    for max_k in [100000, 500000, 1000000, 2000000, 5000000, 10000000]:
        if time.time() > deadline - 30:
            break
        
        try:
            spf, primes = compute_spf(max_k)
        except MemoryError:
            continue
        
        # Precompute sigma for all numbers using SPF
        # sigma(n) = sigma(n / p^a) * (1 + p + ... + p^a) where p^a || n
        sigma_arr = [1] * (max_k + 1)
        for i in range(2, max_k + 1):
            p = spf[i]
            temp = i
            power_sum = 1
            power_val = 1
            while temp % p == 0:
                temp //= p
                power_val *= p
                power_sum += power_val
            sigma_arr[i] = sigma_arr[temp] * power_sum
        
        # Group by M = k * sigma(k)
        # Only process new k values to avoid duplicates in m_to_ks
        for k in range(prev_max_k + 1, max_k + 1):
            if time.time() > deadline - 10:
                break
            m = k * sigma_arr[k]
            if m > 10**18:
                break
            if m not in m_to_ks:
                m_to_ks[m] = []
            m_to_ks[m].append(k)
            
            if len(m_to_ks[m]) > len(best_ks):
                best_M = m
                best_ks = list(m_to_ks[m])
        
        prev_max_k = max_k
        
        if time.time() > deadline - 5:
            break
    
    if len(best_ks) > 3:
        return {'M': int(best_M), 'k': sorted([int(x) for x in best_ks])}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
