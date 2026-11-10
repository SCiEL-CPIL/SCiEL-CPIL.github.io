import numpy as np

def prime_number_theorem_partial_function():
    K = 50000
    
    # Linear Sieve for Mobius
    mu = np.ones(K + 1, dtype=np.float64)
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    mu[1] = 1
    
    k_vals = np.arange(1, K + 1)
    
    # S = sum mu(k)/k
    S = np.sum(mu[1:] / k_vals)
    
    # a_h = -sum mu(k) log(k) / k
    # Note: log(1) = 0, so term for k=1 is 0
    a_h = -np.sum(mu[1:] * np.log(k_vals) / k_vals)
    
    # M_f(x) = 1 for all x
    # M_g(x) = 1 - S * x
    # Max M_g(x) for x in [1, 10K]
    max_x = 10 * K
    
    if S > 0:
        M_max = 1 - S * 1
    else:
        M_max = 1 - S * max_x
        
    # Ensure M_max is at least 1 to avoid division by small number if S is very small positive?
    # If S > 0, M_max < 1. We can scale up.
    # If S < 0, M_max > 1. We must scale down.
    
    c = 1.0 / M_max if M_max > 0 else 1.0
    
    # If M_max <= 1, c >= 1.
    # If M_max > 1, c < 1.
    
    domain = k_vals.astype(np.int64)
    values = c * mu[1:]
    
    return domain, values
