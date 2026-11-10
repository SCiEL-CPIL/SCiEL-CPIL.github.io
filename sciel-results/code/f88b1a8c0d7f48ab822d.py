# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n = 1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    mu[0] = 0
    is_prime[0] = False
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


def _check_feasibility(domain, values, K):
    """Check if M(x) <= 1.0001 for all x in [1, 10K]."""
    max_x = 10 * K
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k, v in zip(domain, values):
        if v != 0:
            g[k::k] += v
    G = np.cumsum(g[1:])  # G[i] = M(i+1)
    return np.max(G) <= 1.0001


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 610  # Safety margin before 630s limit

    # Try different K values, starting large
    for K in [200000, 100000, 50000, 10000, 1000, 100]:
        if time.time() - start_time > deadline:
            break

        mu = _compute_mobius(K)
        nonzero_mask = mu[1:] != 0
        domain = np.where(nonzero_mask)[0] + 1
        values = mu[1:][nonzero_mask].copy()

        # Check feasibility
        if _check_feasibility(domain, values, K):
            return domain, values

    # Fallback to seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
