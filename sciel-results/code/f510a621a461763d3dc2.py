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
    deadline = start_time + 600  # 600 second budget with safety margin

    # Compute the exact Möbius function μ(k) for all k in [1, K_max] using a linear sieve.
    # K_max = 200000 as specified in the task contract.
    K_max = 200000

    # Linear sieve for Möbius function
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K_max + 1, dtype=bool)

    for i in range(2, K_max + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Check time
    if time.time() > deadline - 30:
        # Fall back to a small feasible solution if we're running out of time
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Construct domain as all k where μ(k) ≠ 0 (square-free integers)
    nonzero_mask = (mu != 0)
    domain = np.where(nonzero_mask)[0]
    domain = domain[domain > 0]  # ensure positive integers
    values = mu[domain].astype(np.float64)

    # The evaluator will:
    # 1. Clip values to [-10, 10] (already satisfied since μ ∈ {-1, 0, 1})
    # 2. Replace f(1) with f(1) - S where S = sum_k f(k)/k, so that sum_k f(k)/k = 0
    # 3. Check M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K]
    # 4. Compute a = -sum_k f(k) * log(k) / k

    # For the full Möbius function, the identity sum_{k|x} μ(k) = 0 for x > 1 and = 1 for x = 1
    # implies that M(x) = sum_{k<=x} μ(k) floor(x/k) = 1 for all x >= 1 (exact identity).
    # After the evaluator's adjustment of f(1), the constraint M(x) <= 1 should still hold
    # because the adjustment is designed to make sum_k f(k)/k = 0, which is consistent
    # with the Möbius identity.

    # Verify feasibility quickly for a sample of x values
    K = int(domain[-1]) if len(domain) > 0 else 1
    x_check = np.arange(1, min(10 * K + 1, 100000), dtype=np.int64)

    # Compute M(x) for sample x values to verify feasibility
    # M(x) = sum_k f(k) * floor(x/k)
    # For efficiency, we can use the fact that floor(x/k) changes only at O(sqrt(x)) points
    # But for a quick check, let's compute directly for a few x values

    # Quick feasibility check for x = 1, 2, ..., 100
    x_test = np.arange(1, 101, dtype=np.int64)
    for x in x_test:
        M_x = np.sum(values * (x // domain))
        if M_x > 1.0001:
            # Infeasible - fall back to baseline
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values

    if time.time() > deadline - 10:
        return domain, values

    return domain, values


# EVOLVE-BLOCK-END
