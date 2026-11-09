# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Dirichlet convolution inversion via the Mobius function.
    We compute the exact Mobius function mu(k) for all k <= K, retain only square-free
    indices (where mu(k) != 0), and set f(k) = mu(k). This directly approximates the
    infinite identities:
      sum_k mu(k)/k = 0  (after the evaluator's adjustment at f(1))
      M(x) = sum_k mu(k) * floor(x/k) = 1 for all x >= 1 (in the infinite limit)

    For finite K, M(x) stays close to 1 (with oscillations bounded by the Mertens function),
    and the objective a = -sum_k mu(k) log(k)/k approaches 1 as K grows.

    Returns:
        (domain, values): tuple of two numpy arrays of equal length describing f.
    """
    # Choose K to balance computational cost and objective value.
    # K = 100000 gives a good tradeoff; computation is fast with sieve.
    K = 100000

    # Compute Mobius function mu(k) for all k <= K using a linear sieve.
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

    # Retain only indices where mu(k) != 0 (square-free integers)
    mask = mu != 0
    domain = np.nonzero(mask)[0]
    # domain starts at 1 since mu[1]=1, but we need positive integers only
    domain = domain[domain >= 1]
    values = mu[domain].astype(np.float64)

    # Ensure domain is strictly positive integers
    # (mu[0] is 0 so it's already excluded)

    return domain, values


# EVOLVE-BLOCK-END
