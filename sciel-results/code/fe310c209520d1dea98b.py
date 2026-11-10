import numpy as np

def mobius_sieve(n):
    if n < 1:
        return np.array([], dtype=np.int64)
    mu = np.ones(n+1, dtype=np.int64)
    is_prime = np.ones(n+1, dtype=bool)
    primes = []
    mu[0] = 0
    for i in range(2, n+1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i*p] = False
            if i % p == 0:
                mu[i*p] = 0
                break
            else:
                mu[i*p] = -mu[i]
    return mu

def prime_number_theorem_partial_function():
    K = 100
    mu = mobius_sieve(K)
    domain = np.arange(1, K+1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # Clip to [-10, 10] (mu is already in {-1,0,1}, so no change)
    values_clipped = np.clip(values, -10, 10)
    
    # Compute S = sum f(k)/k
    S = np.sum(values_clipped / domain)
    
    # Adjust f(1) so that sum f(k)/k = 0
    values_adj = values_clipped.copy()
    values_adj[0] = values_adj[0] - S
    
    # Verify feasibility: M(x) <= 1.0001 for x in [1, 10K]
    max_x = 10 * K
    # Efficient computation: M(x) = sum_k f(k) * floor(x/k)
    # For each k, floor(x/k) changes at x = k, 2k, 3k, ...
    # We can compute M(x) for all x by iterating over k and adding f(k) to ranges
    
    M = np.zeros(max_x + 1, dtype=np.float64)
    for i, k in enumerate(domain):
        f_k = values_adj[i]
        if f_k == 0:
            continue
        # floor(x/k) = q for x in [q*k, (q+1)*k - 1]
        # For x in [1, max_x], q ranges from 1 to max_x//k
        q_max = max_x // k
        for q in range(1, q_max + 1):
            start = q * k
            end = min((q+1) * k - 1, max_x)
            if start <= max_x:
                M[start:end+1] += f_k * q
    
    if np.max(M[1:]) > 1.0001:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    a = -np.sum(values_adj * np.log(domain) / domain)
    return domain, values_adj
