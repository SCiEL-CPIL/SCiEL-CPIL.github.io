# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of positive integers to reals
    maximizing a = -sum_k f(k) log(k)/k subject to:
      1) After clipping to [-10,10] and adjusting f(1) so sum f(k)/k = 0,
         M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, 10K].
      2) K <= 200000, m >= 1.

    Strategy: Use Mobius function mu(k) for k=1..K.
    The identity sum_{k=1}^x mu(k) floor(x/k) = 1 holds for infinite sums.
    For finite K, M(x) = 1 - Tail(x). We verify M(x) <= 1.0001.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 30.0

    # Fallback small solution if something goes wrong or time runs out
    def get_fallback():
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # Check time before starting
    if time.time() - start_time > time_limit - safety_margin:
        return get_fallback()

    K = 200000
    N = 10 * K

    # 1. Compute Mobius function mu(k) for k=1..K
    mu = np.zeros(K + 1, dtype=np.int64)
    mu[1] = 1
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
                
    if time.time() - start_time > time_limit - safety_margin:
        return get_fallback()

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)

    # 2. Clip values (no-op for mu, but required by spec logic)
    values_clipped = np.clip(values, -10.0, 10.0)

    # 3. Adjust f(1)
    # S = sum_k f(k)/k
    # Using vectorized division
    S = np.sum(values_clipped / domain)
    values_adj = values_clipped.copy()
    values_adj[0] -= S

    if time.time() - start_time > time_limit - safety_margin:
        return get_fallback()

    # 4. Verify M(x) <= 1.0001
    # M(x) = sum_{j=1}^x g(j) where g(j) = sum_{d|j} f(d)
    # Compute g(j) for j=1..N
    g = np.zeros(N + 1, dtype=np.float64)
    
    # Efficiently compute g using sieve-like loop
    # f is 0-indexed in values_adj, but corresponds to k=1..K
    # values_adj[k-1] is f(k)
    for k in range(1, K + 1):
        f_k = values_adj[k - 1]
        if f_k != 0.0:
            # Add f_k to all multiples of k in g
            # g[k], g[2k], ..., g[floor(N/k)*k]
            g[k::k] += f_k
            
        # Time check
        if k % 10000 == 0:
            if time.time() - start_time > time_limit - safety_margin:
                return get_fallback()

    # M(x) is cumulative sum of g
    # g[0] is unused (index 0 corresponds to j=0, but we start from 1)
    # g[1] corresponds to j=1
    M = np.cumsum(g)
    
    # Check max M(x) for x in [1, N]
    # M[1] corresponds to x=1, M[N] corresponds to x=N
    max_M = np.max(M[1:N+1])
    
    if max_M > 1.0001:
        # Infeasible. 
        # We could try to reduce K, but for now fallback to ensure validity.
        # A smaller K might be feasible. Let's try K=10000 as a secondary fallback?
        # Or just use the small seed.
        return get_fallback()

    if time.time() - start_time > time_limit - safety_margin:
        return get_fallback()

    # 5. Compute metric a
    # a = -sum_k f(k) log(k) / k
    # Using adjusted values
    log_k = np.log(domain)
    a = -np.sum(values_adj * log_k / domain)

    return domain, values

# EVOLVE-BLOCK-END
