def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    # Safety margin: 50s out of 60s limit
    deadline = start_time + 50
    
    N = 1000000
    limit = N + 1  # We need lpf up to N+1
    
    # 1. Sieve of Eratosthenes using NumPy
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # Check time
    if time.time() > deadline:
        # Fallback: return a known good candidate or random?
        # Since the sieve is fast, this shouldn't happen.
        # But if it does, we need a valid n.
        # A safe guess: 10000 (LPF(10000)=5, LPF(10001)=73? 10001=73*137. Max=137)
        # Better to return a smooth number if possible.
        # 1024 is 2^10. 1025=5^2*41. Max=41.
        # 10000 is 10^4 = 2^4*5^4. LPF=5.
        # 10001 = 73 * 137. LPF=137.
        # Let's just return 10000 to be safe? No, we want low LPF.
        # 1024, 1025 -> 41.
        # 1023=3*11*31, 1024=2^10 -> 31.
        # 1024 is good.
        return 1024

    primes = np.where(is_prime)[0]
    
    # 2. Largest Prime Factor Sieve using NumPy
    lpf = np.zeros(limit + 1, dtype=np.int32)
    for p in primes:
        if time.time() > deadline:
            break
        lpf[p::p] = p
        
    # 3. Vectorized search for best n
    # lpf[n] for n in [10000, N] and lpf[n+1] for n in [10000, N]
    lpf_n = lpf[10000:N+1]      # LPF of n for n=10000..1000000
    lpf_np1 = lpf[10001:N+2]    # LPF of n+1 for n=10000..1000000
    
    scores = np.maximum(lpf_n, lpf_np1)
    best_idx = np.argmin(scores)
    best_n = 10000 + int(best_idx)
    
    return best_n
