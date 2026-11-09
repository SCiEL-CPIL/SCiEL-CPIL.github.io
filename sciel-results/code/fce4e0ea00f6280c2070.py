import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    K = 100
    
    # Compute Mobius function using linear sieve
    mu = np.ones(K + 1, dtype=np.int64)
    is_composite = np.zeros(K + 1, dtype=bool)
    primes = []
    
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
    
    # domain: 1 to K, values: mu(1) to mu(K)
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K + 1].astype(np.float64)
    
    return domain, values
