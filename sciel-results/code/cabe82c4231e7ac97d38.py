# EVOLVE-BLOCK-START
import numpy as np


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
    # Limit for the domain
    K = 100000
    
    # Compute the Mobius function mu(n) for n = 1..K using a linear sieve
    mu = np.zeros(K + 1, dtype=np.int64)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Build domain: all k in [1, K] where mu(k) != 0
    mask = (mu[1:K + 1] != 0)
    domain = np.where(mask)[0] + 1  # 1-indexed
    
    # Compute f(1) = -sum_{k=2}^{K} mu(k)/k
    k_arr = np.arange(2, K + 1, dtype=np.float64)
    mu_k = mu[2:K + 1].astype(np.float64)
    sum_mu_over_k = np.sum(mu_k / k_arr)
    f1 = -sum_mu_over_k
    
    # Build values
    values = np.empty(len(domain), dtype=np.float64)
    for i, k in enumerate(domain):
        if k == 1:
            values[i] = f1
        else:
            values[i] = float(mu[k])
    
    # Clip values to [-10, 10]
    values = np.clip(values, -10.0, 10.0)
    
    return domain.astype(np.int64), values


# EVOLVE-BLOCK-END
