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
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave a safety margin before the 600s limit

    # First, compute the Möbius function up to a reasonable bound using the sieve.
    # We'll use a bound of 100000 to start, and adjust if needed.
    K_max = 100000
    mu = np.ones(K_max + 1, dtype=np.int64)
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

    # The ideal function is f(k) = mu(k), which gives M(x) = 1 for all x >= 1.
    # The objective a = -sum f(k) log(k)/k. For mu, this converges to 1 as K -> infinity.
    # However, the constraint is that M(x) <= 1.0001 for x in [1, 10K].
    # For mu, M(x) = 1 for all x >= 1, so it satisfies M(x) <= 1.0001.
    # The only issue is that the domain must be finite and the values must be clipped to [-10, 10].
    # mu(k) is always in {-1, 0, 1}, so clipping doesn't change it.
    # We want to maximize -sum mu(k) log(k)/k over a finite domain.
    # The partial sum -sum_{k=1}^{K} mu(k) log(k)/k approaches 1 from above (oscillating).
    # We want to choose a finite set of k to include in the domain such that the sum is maximized.
    # Since mu(k) = 0 for k with a squared prime factor, we only need to include k where mu(k) != 0.
    # The contribution of k is -mu(k) log(k)/k.
    # For mu(k) = 1, contribution is -log(k)/k < 0.
    # For mu(k) = -1, contribution is log(k)/k > 0.
    # So we want to include k with mu(k) = -1 (positive contribution) and exclude k with mu(k) = 1 (negative contribution).
    # But we must be careful: excluding some k might violate the M(x) constraint.
    # Let's first check: if we include all k with mu(k) = -1 and exclude all k with mu(k) = 1, what happens to M(x)?
    # M(x) = sum_{k in domain} mu(k) floor(x/k).
    # If we only include k with mu(k) = -1, then M(x) = -sum_{k in domain, mu(k)=-1} floor(x/k).
    # This is negative for all x, so M(x) <= 1 is satisfied.
    # But we also need to handle f(1). The evaluator sets f(1) = f(1) - S where S = sum f(k)/k.
    # Let's think about this more carefully.

    # Alternative approach: Use a subset of the Möbius function that maximizes the objective.
    # Let's try including k up to K where the partial sum is high.
    # Actually, the partial sum -sum_{k=1}^{K} mu(k) log(k)/k oscillates around 1.
    # We want to find K where it's at a local maximum above 1.
    
    # Compute the partial sum -sum_{k=1}^{K} mu(k) log(k)/k for K up to K_max.
    # log_k_over_k = log(k)/k for k >= 1
    log_k_over_k = np.zeros(K_max + 1, dtype=np.float64)
    for k in range(2, K_max + 1):
        log_k_over_k[k] = np.log(k) / k
    # log(1)/1 = 0
    
    # partial_sum[K] = -sum_{k=1}^{K} mu(k) * log(k)/k
    partial_sum = np.zeros(K_max + 1, dtype=np.float64)
    for k in range(1, K_max + 1):
        partial_sum[k] = partial_sum[k-1] - mu[k] * log_k_over_k[k]
    
    # Find the K in [1, K_max] where partial_sum[K] is maximized.
    best_K = int(np.argmax(partial_sum[1:K_max+1]) + 1)
    best_val = partial_sum[best_K]
    
    # But wait: we need to verify that M(x) <= 1.0001 for all x in [1, 10*best_K].
    # For the full Möbius function up to best_K, M(x) = sum_{k=1}^{best_K} mu(k) floor(x/k).
    # The classic identity says sum_{k=1}^{infinity} mu(k) floor(x/k) = 1 for x >= 1.
    # The partial sum up to best_K is 1 - sum_{k=best_K+1}^{infinity} mu(k) floor(x/k).
    # The tail sum might make M(x) slightly above or below 1.
    # We need to check numerically.
    
    # Let's compute M(x) for x in [1, 10*best_K] and check if any M(x) > 1.0001.
    # This is O(best_K * 10*best_K) which is too slow for best_K ~ 100000.
    # We need a smarter approach.
    
    # Alternative: Use the fact that M(x) = sum_{k} mu(k) floor(x/k).
    # For x in [1, 10*best_K], floor(x/k) = 0 for k > x.
    # So M(x) = sum_{k=1}^{min(x, best_K)} mu(k) floor(x/k).
    # We can compute this for all x in [1, 10*best_K] in O(best_K * log(best_K)) using a sieve-like approach.
    
    # But 10*best_K could be up to 10^6, and best_K up to 10^5.
    # O(10^6 * log(10^6)) ~ 4e7, which is feasible.
    
    # Let's implement this check.
    def check_M_constraint(K, mu_vals, log_k_over_k_vals):
        """Check if M(x) <= 1.0001 for all x in [1, 10*K]."""
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        for k in range(1, K + 1):
            if mu_vals[k] != 0:
                M[k:max_x+1:k] += mu_vals[k]
        return np.max(M[1:max_x+1]) <= 1.0001

    # Check for the best_K.
    if check_M_constraint(best_K, mu, log_k_over_k):
        domain = np.array([k for k in range(1, best_K + 1) if mu[k] != 0], dtype=np.int64)
        values = mu[domain].astype(np.float64)
        return domain, values

    # If the constraint is violated, try smaller K values.
    # Find the largest K <= best_K such that the constraint is satisfied and the objective is high.
    for K in range(best_K, 0, -1):
        if K <= 0:
            break
        if time.time() > deadline:
            break
        if check_M_constraint(K, mu, log_k_over_k):
            val = partial_sum[K]
            domain = np.array([k for k in range(1, K + 1) if mu[k] != 0], dtype=np.int64)
            values = mu[domain].astype(np.float64)
            return domain, values

    # Fallback to the original small domain if time runs out or no K works.
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END

