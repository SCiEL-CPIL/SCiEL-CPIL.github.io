import numpy as np
import time

def prime_number_theorem_partial_function():
    # Deadline guard
    start_time = time.time()
    
    # Choose K
    # K=5000 is a reasonable size. 
    # Computing mu up to 5000 is very fast.
    K = 5000
    
    # Compute Mobius function mu(k) for k=1..K
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    # mu[1] should be 1
    mu[1] = 1
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    return domain, values
