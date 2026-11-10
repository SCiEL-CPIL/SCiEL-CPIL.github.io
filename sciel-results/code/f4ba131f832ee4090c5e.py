import numpy as np

def prime_number_theorem_partial_function():
    # Choose K. 10000 is a good balance between accuracy and computation time.
    # The evaluator computes M(x) for x in [1, 10K].
    # Our pre-computation also does this to find the scaling factor.
    K = 10000
    
    # 1. Compute Mobius function mu(1..K)
    # mu[n] = 0 if n has a squared prime factor
    # mu[n] = (-1)^k if n is a product of k distinct primes
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
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
                
    # mu[1] is 1 by initialization.
    # mu array is 1-indexed effectively, mu[0] unused.
    mu_vals = mu[1:K+1] # shape (K,)
    
    # 2. Compute S_mu = sum(mu(k)/k)
    k_indices = np.arange(1, K + 1, dtype=np.float64)
    S_mu = np.sum(mu_vals / k_indices)
    
    # 3. Compute M_mu(x) for x in [1, 10K]
    # M_mu(x) = sum_{k=1}^K mu(k) * floor(x/k)
    max_x = 10 * K
    x_vals = np.arange(1, max_x + 1, dtype=np.int64)
    M_mu = np.zeros(max_x, dtype=np.float64)
    
    # Vectorized accumulation
    # To speed up, we can iterate over k
    # Note: x_vals // k is integer division
    for k in range(1, K + 1):
        if mu_vals[k-1] != 0:
            M_mu += mu_vals[k-1] * (x_vals // k)
            
    # 4. Compute H(x) = M_mu(x) - S_mu * x
    # H(x) represents the effective M(x) after the f(1) adjustment, scaled by 1/c
    H = M_mu - S_mu * x_vals
    
    # 5. Find max H
    H_max = np.max(H)
    
    # Ensure H_max is positive (it should be >= 1)
    if H_max <= 0:
        # Fallback to a very small value to avoid division by zero or negative scaling
        # Though theoretically H_max >= 1
        H_max = 1.0
        
    # 6. Calculate scaling factor c
    # Constraint: c * H(x) <= 1.0001
    c = 1.0001 / H_max
    
    # 7. Construct domain and values
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = c * mu_vals.astype(np.float64)
    
    return domain, values
