import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    MAX_N = 1000001
    
    # Precompute smallest prime factor (SPF)
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute p_pow[i] and rem[i]
    # p_pow[i] is the highest power of spf[i] that divides i
    # rem[i] is i // p_pow[i]
    p_pow = [1] * (MAX_N + 1)
    rem = [1] * (MAX_N + 1)
    
    p_pow[1] = 1
    rem[1] = 1
    
    # pp[i] will store the powerful part of i
    pp = [1] * (MAX_N + 1)
    pp[1] = 1
    
    for i in range(2, MAX_N + 1):
        p = spf[i]
        j = i // p
        
        if j % p == 0:
            # The exponent of p in i is at least 2
            # p_pow[j] is the highest power of p dividing j
            # So p_pow[i] = p_pow[j] * p
            p_pow[i] = p_pow[j] * p
        else:
            # The exponent of p in i is 1
            p_pow[i] = p
            
        rem[i] = i // p_pow[i]
        
        # Compute powerful part
        # If the exponent of p in i is >= 2, i.e., p_pow[i] >= p^2
        # Note: p_pow[i] is a power of p.
        # We can check if p_pow[i] // p % p == 0? No.
        # Just check if p_pow[i] >= p * p.
        if p_pow[i] >= p * p:
            pp[i] = p_pow[i] * pp[rem[i]]
        else:
            pp[i] = pp[rem[i]]
            
        # Check time periodically
        if i % 100000 == 0 and time.time() > deadline:
            # If we run out of time, we can't compute pp for higher numbers.
            # But we need pp up to 1000001.
            # Given N=1e6, this should be very fast.
            pass
    
    # Find the best n in [10000, 1000000]
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    for n in range(10000, MAX_N):
        val = pp[n] * pp[n+1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if time.time() > deadline:
            break
            
    return best_n
