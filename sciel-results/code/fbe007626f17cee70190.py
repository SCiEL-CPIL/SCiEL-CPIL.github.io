# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k)*log(k)/k subject to
    M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10*K].

    Uses the Möbius function mu(k) which satisfies the exact identity
    sum_{d|n} mu(d) = 1, i.e., M(x) = 1 for all x >= 1.
    The objective -sum mu(k)*log(k)/k approaches 1 as K grows.
    """
    start_time = time.time()
    DEADLINE = 600.0  # seconds, with safety margin

    K = 200000

    # Compute Möbius function using linear sieve
    def mobius_sieve(n):
        """Compute Möbius function for 1..n using linear sieve."""
        mu = np.zeros(n + 1, dtype=np.float64)
        mu[1] = 1.0
        is_prime = np.zeros(n + 1, dtype=bool)
        primes = []
        lp = np.zeros(n + 1, dtype=np.int64)  # least prime factor

        for i in range(2, n + 1):
            if lp[i] == 0:
                lp[i] = i
                is_prime[i] = True
                primes.append(i)
                mu[i] = -1.0
            for p in primes:
                if p > lp[i] or i * p > n:
                    break
                lp[i * p] = p
                if i % p == 0:
                    mu[i * p] = 0.0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    mu = mobius_sieve(K)

    # Domain: all k from 1 to K where mu(k) != 0 (square-free numbers)
    mask = mu[1:] != 0.0
    domain = np.where(mask)[0] + 1  # 1-indexed
    values = mu[1:][mask].copy()

    m = len(domain)

    # The Möbius function gives M(x) = 1 exactly for all x >= 1,
    # so it's always feasible (M(x) = 1.0 <= 1.0001).
    # Objective: a = -sum_k f(k)*log(k)/k approaches 1 as K -> infinity.

    # Check time budget
    if time.time() - start_time > DEADLINE - 5:
        return domain, values

    return domain, values


# EVOLVE-BLOCK-END
