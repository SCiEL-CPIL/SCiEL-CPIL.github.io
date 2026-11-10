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
    # Larger K gives better score (closer to 1) and is more likely to satisfy M(x) <= 1.0001
    # because the error term R(x) = sum_{k=K+1}^x mu(k) floor(x/k) is smaller.
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try a range of K values, prioritizing larger K for better score
    # Cost is O(K^2). K=200000 -> 4e11 ops. Might take ~50-100s.
    # We have 550s budget.
    K_candidates = [200000, 150000, 100000, 80000, 60000, 40000, 20000, 10000, 5000, 2000, 1000]
    
    for K in K_candidates:
        if time.time() > deadline - 60:  # Leave 60s for finalization
            break
        if K > K_max:
            continue
            
        S1 = np.sum(mu_over_k[:K])
        if S1 < 0:
            # If S1 < 0, f(1) > 1. This is allowed, but let's check feasibility first.
            # Actually, S1 = sum mu(k)/k. For large K, S1 is close to 0.
            # It oscillates around 0.
            pass

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
        
        # Optimized loop
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
                # Since we are trying larger K first, and score increases with K,
                # we can break if we find a valid K?
                # No, larger K is always better if valid.
                # So we can break after the first valid K found (which is the largest).
                break
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
