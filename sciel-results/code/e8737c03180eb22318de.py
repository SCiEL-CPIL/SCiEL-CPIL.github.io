# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(n):
    """Compute Mobius function mu(k) for k = 1..n using sieve."""
    mu = np.ones(n + 1, dtype=np.int8)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    mu[1] = 1
    return mu


def _evaluate(domain, values, K):
    """Simulate the evaluator and return (feasible, score)."""
    # Clip values
    clipped = np.clip(values, -10, 10)
    
    # Compute S = sum_k f(k)/k
    k_float = domain.astype(np.float64)
    S = np.sum(clipped / k_float)
    
    # Adjust f(1)
    if 1 in domain:
        idx1 = np.where(domain == 1)[0][0]
        adjusted = clipped.copy()
        adjusted[idx1] = adjusted[idx1] - S
        domain_adj = domain
    else:
        adjusted = np.append(clipped, -S)
        domain_adj = np.append(domain, 1)
    
    # Check feasibility: M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, 10K]
    max_x = 10 * K
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k, v in zip(domain_adj, adjusted):
        if v != 0:
            multiples = np.arange(k, max_x + 1, k)
            g[multiples] += v
    
    M = np.cumsum(g)
    max_M = np.max(M[1:max_x+1])
    
    if max_M > 1.0001:
        return False, -np.inf
    
    # Compute objective
    k_float = domain_adj.astype(np.float64)
    score = -np.sum(adjusted * np.log(k_float) / k_float)
    
    return True, score


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    for K in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() > deadline - 30:
            break
        
        mu = _compute_mobius(K)
        # Only include k where mu(k) != 0 (squarefree numbers)
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values = mu[1:][mask].astype(np.float64)
        
        feasible, score = _evaluate(domain, values, K)
        if feasible and score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()
    
    if best_domain is None:
        # Fallback to original
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values


# EVOLVE-BLOCK-END
