# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.ones(N + 1, dtype=np.int8)
    primes = []
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    mu[1] = 1
    return mu

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    t_start = time.time()
    time_budget = 600
    safety_margin = 30

    # Use a large K to approximate the infinite sum well
    K = 200000
    
    # Check time
    if time.time() - t_start > time_budget - safety_margin:
        # Fallback if somehow too late (should not happen)
        domain = np.array([1], dtype=np.int64)
        values = np.array([0.0], dtype=np.float64)
        return domain, values

    # Compute Moebius function
    mu = _compute_mobius(K)
    
    # Set scaling factor c
    # We determined c = -10 maximizes the objective a = -c * L_mu
    # where L_mu = sum mu(k) log(k) / k^2 > 0
    # and M(x) <= 1.0001 is satisfied for c = -10
    c = -10.0
    
    # Construct domain and values
    indices = np.arange(1, K + 1, dtype=np.float64)
    values = c * mu[1:].astype(np.float64) / indices
    
    # Clip values to [-10, 10] just in case, though they should be within bounds
    values = np.clip(values, -10.0, 10.0)
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    return domain, values

# EVOLVE-BLOCK-END
