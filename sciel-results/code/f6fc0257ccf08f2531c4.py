import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We will search k up to K_MAX
    # Let's start with 10^7 and see if we have time to go higher?
    # Or just pick a fixed reasonable limit like 5*10^7 or 10^8.
    # 10^8 is likely safe for time.
    K_MAX = 10**8
    
    # Sieve of Eratosthenes to find smallest prime factor (SPF)
    # spf[i] will store the smallest prime factor of i
    spf = list(range(K_MAX + 1))
    for i in range(2, int(math.isqrt(K_MAX)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, K_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Dictionary to store counts of M = k * sigma(k)
    # key: M, value: list of k's
    m_to_k = {}
    
    # We need to compute sigma(k) for all k.
    # sigma is multiplicative.
    # We can compute sigma(k) using spf.
    # sigma(k) = sigma(m) * (1 + p + ... + p^a) where k = m * p^a and p does not divide m.
    
    sigma = [0] * (K_MAX + 1)
    sigma[1] = 1
    
    for k in range(2, K_MAX + 1):
        p = spf[k]
        # Find the highest power of p dividing k
        # Let k = m * p^a
        m = k
        a = 0
        while m % p == 0:
            m //= p
            a += 1
        
        # sigma(k) = sigma(m) * (p^(a+1) - 1) / (p - 1)
        # We need sigma(m). m < k, so it is already computed.
        # But wait, we are iterating k from 2 upwards.
        # m is a divisor of k, so m < k (unless a=0, but p=spf[k] so a>=1).
        # So sigma[m] is available.
        
        # Calculate geometric series sum: 1 + p + ... + p^a
        # sum = (p^(a+1) - 1) // (p - 1)
        # Be careful with overflow? Python handles big ints.
        # But p^(a+1) might be large if p is large and a is large?
        # Max k is 10^8. Max p is 10^8 (if k is prime). Then a=1.
        # p^2 = 10^16. Fits in Python int easily.
        
        sum_powers = 0
        power = 1
        for _ in range(a + 1):
            sum_powers += power
            power *= p
            
        sigma[k] = sigma[m] * sum_powers
        
        M = k * sigma[k]
        
        if M <= 10**18:
            if M in m_to_k:
                m_to_k[M].append(k)
            else:
                m_to_k[M] = [k]
        
        # Time check
        if k % 1000000 == 0 and time.time() > deadline:
            break
            
    # Find the M with the most k's
    best_M = 0
    best_ks = []
    
    for M, ks in m_to_k.items():
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
            
    # If we didn't find anything better than 3, we might want to fall back to the known solution?
    # Or just return what we found. The search should find at least the known ones if K_MAX is large enough.
    # 336 is small, so it will be found.
    
    # Ensure types are correct
    # M is int, best_ks is list of ints.
    
    if not best_ks:
        # Fallback to seed solution if search failed completely
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_ks}
