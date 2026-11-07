import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # We can dynamically adjust N based on time, but a fixed large N is simpler.
    # Let's try N = 20_000_000 first.
    N = 20_000_000
    
    # 1. SPF Sieve
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute sigma and group
    groups = {}
    limit = 10**18
    
    # Optimization: local variables for speed
    spf_local = spf
    groups_setdefault = groups.setdefault
    groups_get = groups.get
    
    for k in range(1, N + 1):
        # Check time
        if k % 1000000 == 0:
            if time.time() > deadline:
                break
        
        # Compute sigma(k)
        n = k
        sigma_k = 1
        while n > 1:
            p = spf_local[n]
            power_sum = 1
            power_val = 1
            while n % p == 0:
                n //= p
                power_val *= p
                power_sum += power_val
            sigma_k *= power_sum
        
        M = k * sigma_k
        if M <= limit:
            if M in groups:
                groups[M].append(k)
            else:
                groups[M] = [k]
        else:
            # Since k is increasing, M is generally increasing.
            # However, sigma(k)/k can drop.
            # But for k > sqrt(limit), M >= k^2 > limit.
            # So we can break if k > sqrt(limit).
            # sqrt(10^18) = 10^9.
            # So we won't break early based on M > limit until k is huge.
            pass

    # 3. Find best M
    best_M = 0
    best_k_list = []
    max_count = 0
    
    for M, k_list in groups.items():
        if len(k_list) > max_count:
            max_count = len(k_list)
            best_M = M
            best_k_list = k_list

    return {'M': best_M, 'k': best_k_list}
