def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 1000000
    limit = N + 1  # We need lpf up to N+1
    
    # 1. Sieve of Eratosthenes using NumPy
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Largest Prime Factor Sieve using NumPy
    lpf = np.zeros(limit + 1, dtype=np.int32)
    for p in primes:
        lpf[p::p] = p
    
    # 3. Search for best n
    best_n = 10000
    best_score = np.iinfo(np.int32).max
    
    # Convert to lists for faster scalar access, or use vectorized ops
    lpf_list = lpf.tolist()
    
    for n in range(10000, N + 1):
        if (n - 10000) % 200000 == 0:
            if time.time() > deadline:
                break
        
        p_n = lpf_list[n]
        p_np1 = lpf_list[n + 1]
        score = p_n if p_n > p_np1 else p_np1
        
        if score < best_score:
            best_score = score
            best_n = n
            if best_score <= 3:
                # Can't do better than 3 (need both n and n+1 to have only prime factors 2,3)
                # n=2^a*3^b, n+1=2^c*3^d
                # This is very restrictive. Let's continue but we know 3 is likely the min.
                pass
    
    return best_n
