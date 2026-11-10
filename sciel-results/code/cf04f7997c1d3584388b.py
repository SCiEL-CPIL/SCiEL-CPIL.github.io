import numpy as np
import time
import math

def _mobius_sieve(K):
    """Compute the Möbius function mu(n) for n = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    is_prime[0] = False
    if K >= 1:
        is_prime[1] = False
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1.0
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def _check_feasibility(mu, K):
    """
    Check if f(k) = mu(k) for k=1..K satisfies M(x) <= 1.0001 for all x in [1, 10K].
    Note: The evaluator normalizes f(1) so that sum f(k)/k = 0.
    Let S = sum_{k=1}^K mu(k)/k.
    f^*(1) = mu(1) - S.
    f^*(k) = mu(k) for k > 1.
    M^*(x) = f^*(1) * x + sum_{k=2}^K mu(k) * floor(x/k)
           = (mu(1) - S) * x + sum_{k=2}^K mu(k) * floor(x/k)
           = mu(1)*x - S*x + sum_{k=2}^K mu(k) * floor(x/k)
    
    We know sum_{k=1}^x mu(k) floor(x/k) = 1 for x >= 1 (if sum goes to infinity).
    For finite K, let M_mu(x) = sum_{k=1}^K mu(k) floor(x/k).
    Then sum_{k=2}^K mu(k) floor(x/k) = M_mu(x) - mu(1) floor(x/1) = M_mu(x) - x.
    
    So M^*(x) = mu(1)*x - S*x + M_mu(x) - x
              = x - S*x + M_mu(x) - x   (since mu(1)=1)
              = M_mu(x) - S*x.
    
    We need M_mu(x) - S*x <= 1.0001 for all x in [1, 10K].
    """
    max_x = 10 * K
    # Compute M_mu(x) for x in 1..10K
    # M_mu(x) = sum_{k=1}^K mu(k) floor(x/k)
    # Use the g(j) method
    g = np.zeros(max_x + 1, dtype=np.float64)
    # mu[1] = 1
    # for k in 1..K: g[k::k] += mu[k]
    # This is O(K log K)
    for k in range(1, K + 1):
        fk = mu[k]
        if fk != 0.0:
            g[k::k] += fk
            
    M_mu = np.cumsum(g[1:])
    
    # Compute S = sum_{k=1}^K mu(k)/k
    # S is a scalar
    # We can compute it efficiently
    # S = sum mu[k]/k for k=1..K
    # Using numpy: np.sum(mu[1:K+1] / np.arange(1, K+1))
    k_range = np.arange(1, K + 1, dtype=np.float64)
    S = np.sum(mu[1:K+1] / k_range)
    
    # Check constraint: M_mu(x) - S*x <= 1.0001
    # x = 1..10K
    x_range = np.arange(1, max_x + 1, dtype=np.float64)
    M_star = M_mu - S * x_range
    
    max_M_star = np.max(M_star)
    
    return max_M_star <= 1.0001, max_M_star

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    # We want the largest K such that f(k)=mu(k) is feasible.
    # Try K = 200,000 first.
    K_max = 200000
    
    # Compute mu up to K_max
    mu = _mobius_sieve(K_max)
    
    # Check feasibility for K_max
    feasible, max_M = _check_feasibility(mu, K_max)
    
    if feasible:
        domain = np.arange(1, K_max + 1, dtype=np.int64)
        values = mu[1:K_max+1].copy()
        return domain, values
    
    # If not feasible, binary search for the largest feasible K
    lo = 1
    hi = K_max
    best_K = lo
    
    while lo <= hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        feasible, _ = _check_feasibility(mu, mid)
        if feasible:
            best_K = mid
            lo = mid + 1
        else:
            hi = mid - 1
            
    domain = np.arange(1, best_K + 1, dtype=np.int64)
    values = mu[1:best_K+1].copy()
    
    return domain, values
