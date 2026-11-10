import numpy as np

def construct():
    N = 1000001
    # Sieve of Eratosthenes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = False
            
    primes = np.where(is_prime)[0]
    
    max_p = np.zeros(N + 1, dtype=np.int64)
    for p in primes:
        max_p[p::p] = p
        
    # Scan for minimum
    # n ranges from 10000 to 1000000
    # max_p[n] and max_p[n+1]
    # indices: 10000 to 1000000 for n
    # indices: 10001 to 1000001 for n+1
    
    n_start = 10000
    n_end = 1000000
    
    pn = max_p[n_start:n_end+1]
    pn1 = max_p[n_start+1:n_end+2]
    
    scores = np.maximum(pn, pn1)
    min_score = np.min(scores)
    best_idx = np.argmin(scores)
    best_n = n_start + best_idx
    
    return int(best_n)
