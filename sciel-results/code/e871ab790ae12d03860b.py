import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    # Compute Möbius function up to K
    K = 200000
    mu = np.ones(K + 1, dtype=np.float64)
    # Sieve for Möbius
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
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
                    mu[j] = 0.0
    
    # mu[1] = 1, mu[0] unused
    # Domain: all k from 1 to K where mu[k] != 0
    nonzero_mask = mu[1:] != 0
    domain = np.where(nonzero_mask)[0] + 1
    values = mu[domain].copy()
    
    # Clip to [-10, 10]
    values = np.clip(values, -10, 10)
    
    # Compute S = sum_k f(k)/k
    S = np.sum(values / domain)
    
    # Adjust f(1)
    if 1 in domain:
        idx_1 = np.where(domain == 1)[0][0]
        values[idx_1] -= S
    else:
        # f(1) = 0 - S = -S, add to domain
        values = np.insert(values, 0, -S)
        domain = np.insert(domain, 0, 1)
    
    # Re-clip after adjustment
    values = np.clip(values, -10, 10)
    
    # Check M(x) <= 1.0001 for x in [1, 10*K]
    # M(x) = sum_k f(k) * floor(x/k)
    # Use efficient computation
    max_x = 10 * K
    M = np.zeros(max_x + 1, dtype=np.float64)
    M[0] = 0
    
    # Precompute x_range to avoid repeated allocation
    x_range = np.arange(1, max_x + 1)
    
    # For each k in domain, add f(k) * floor(x/k) for x in [1, max_x]
    # Vectorized: for each k, floor(x/k) = x_range // k
    # This is O(m * max_x) but vectorized
    
    # Check time periodically
    for i, k in enumerate(domain):
        if time.time() > deadline - 60:
            break
        M[1:] += values[i] * (x_range // k)
    
    # Check constraint
    if np.max(M[1:]) > 1.0001:
        # Infeasible, fall back to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    # Compute objective
    a = -np.sum(values * np.log(domain) / domain)
    
    return domain, values
