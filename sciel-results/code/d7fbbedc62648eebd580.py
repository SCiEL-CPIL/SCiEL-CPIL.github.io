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
    
    # We'll search k up to a limit, adapting based on time
    # sigma(k) is roughly k * log(log(k)) for large k
    # So M = k*sigma(k) ~ k^2 * log(log(k))
    # For M <= 10^18, k can be up to ~10^9 / sqrt(log(log(10^9)))
    # But we need to compute sigma efficiently
    
    # Use a linear sieve for SPF
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
    
    # Try progressively larger max_k
    for max_k in [100000, 500000, 1000000, 2000000, 5000000, 10000000]:
        if time.time() > deadline - 30:  # Need time for final processing
            break
        
        try:
            spf, primes = compute_spf(max_k)
        except MemoryError:
            continue
        
        # Precompute sigma for all numbers using multiplicative property
        # sigma(n) = product over prime factors p^a || n of (p^(a+1)-1)/(p-1)
        sigma_arr = [0] * (max_k + 1)
        sigma_arr[0] = 0
        sigma_arr[1] = 1
        
        # For each prime, update sigma for its multiples
        for p in primes:
            for j in range(p, max_k + 1, p):
                # Factor out all p's from j
                temp = j
                power_sum = 1
                power_val = 1
                while temp % p == 0:
                    temp //= p
                    power_val *= p
                    power_sum += power_val
                # sigma(j) = sigma(temp) * power_sum
                # But we need to be careful - this approach doesn't work directly
                # because temp might still have other prime factors not processed yet
                pass
        
        # Better approach: use the SPF to compute sigma incrementally
        # sigma(n) = sigma(n / p^a) * (1 + p + p^2 + ... + p^a) where p^a is the highest power of p dividing n
        sigma_arr = [1] * (max_k + 1)
        for i in range(2, max_k + 1):
            p = spf[i]
            # Divide i by p until not divisible
            temp = i
            power_sum = 1
            power_val = 1
            while temp % p == 0:
                temp //= p
                power_val *= p
                power_sum += power_val
            sigma_arr[i] = sigma_arr[temp] * power_sum
        
        # Now group by M = k * sigma(k)
        for k in range(1, max_k + 1):
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
        
        if time.time() > deadline - 5:
            break
    
    if len(best_ks) > 3:
        return {'M': int(best_M), 'k': sorted([int(x) for x in best_ks])}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
