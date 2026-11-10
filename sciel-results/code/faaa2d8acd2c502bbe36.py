# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 550  # Leave 80s margin for evaluator overhead
    
    # Sieve for Mobius function up to K_max
    K_max = 200000
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K_max + 1, dtype=bool)
    primes = []
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Precompute mu(k)/k
    k_vals = np.arange(1, K_max + 1)
    mu_over_k = mu[1:] / k_vals.astype(np.float64)
    
    # Precompute log(k)/k for score calculation
    log_over_k = np.log(k_vals) / k_vals.astype(np.float64)
    
    # Try K values from large to small
    # M(x) computation cost is ~K * 10K = 10*K^2 operations
    # K=20000 -> 4e9 ops ~ too slow in Python loop
    # K=10000 -> 1e9 ops ~ ~3s with optimized loop
    # K=5000 -> 2.5e8 ops ~ ~0.5s
    # K=3000 -> 9e7 ops ~ ~0.2s
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try a range of K values, prioritizing larger K for better score
    # a ≈ 1 - O(1/K), so larger K gives better score
    K_candidates = [20000, 15000, 12000, 10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 1000]
    
    for K in K_candidates:
        if time.time() > deadline:
            break
        if K > K_max:
            continue
            
        S1 = np.sum(mu_over_k[:K])
        if S1 < 0:
            continue
            
        # Construct f
        # f(1) = 1 - S1
        # f(k) = mu(k) for k > 1
        domain = k_vals[:K].copy()
        values = mu[1:K+1].astype(np.float64).copy()
        values[0] = 1.0 - S1
        
        # Check constraint M'(x) <= 1 for x in [1, 10K]
        # M'(x) = sum_{k=1}^K f(k) * floor(x/k)
        x_max = 10 * K
        
        # Pre-allocate arrays
        M_prime = np.zeros(x_max, dtype=np.float64)
        x_range = np.arange(1, x_max + 1, dtype=np.int64)
        
        # Optimized loop: pre-allocate x_range, use in-place addition
        for k in range(1, K + 1):
            v = values[k-1]
            if v != 0.0:
                M_prime += v * (x_range // k)
            
        if np.max(M_prime) <= 1.0001:
            # Calculate score
            # a = -sum f(k) log(k) / k
            score = -np.sum(values * log_over_k[:K])
            if score > best_score:
                best_score = score
                best_domain = domain
                best_values = values
        else:
            # Constraint failed
            pass

    if best_domain is None:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END
