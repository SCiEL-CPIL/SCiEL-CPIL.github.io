# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Use the exact Mobius function mu(k) for k up to K.
    The Mobius function satisfies:
      1. sum_{k=1}^infty mu(k)/k = 0 (in the limit, and partial sums are small)
      2. M(x) = sum_{k=1}^x mu(k) floor(x/k) = 1 for all x >= 1 (exact identity)
      3. -sum_{k=1}^infty mu(k) log(k)/k = 1 (in the limit)

    We compute mu(k) for k <= K, keep only square-free k (where mu(k) != 0),
    and set f(k) = mu(k). Then we adjust f(1) to satisfy sum f(k)/k = 0 exactly.

    The metric is a = -sum f(k) log(k)/k, which approaches 1 as K -> infinity.
    """
    # Choose K to balance computation time and accuracy.
    # K = 100000 gives a good approximation and runs fast.
    K = 100000

    # Compute the Mobius function mu(k) for k = 1..K using a sieve.
    mu = np.ones(K + 1, dtype=np.float64)
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            # Sieve out multiples
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            # For squares, mu becomes 0
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0.0

    # Keep only indices where mu(k) != 0 (square-free integers)
    nonzero_mask = mu[1:] != 0.0
    domain = np.nonzero(nonzero_mask)[0] + 1  # 1-based indices
    values = mu[domain].copy()

    # Ensure domain is sorted (it already is from np.nonzero)
    # domain and values are now aligned

    # Clip values to [-10, 10] as per evaluator (though mu is in {-1, 0, 1}, so no change)
    values = np.clip(values, -10.0, 10.0)

    # Adjust f(1) so that sum_k f(k)/k = 0
    # S = sum_k f(k)/k
    # We need f(1) - S = 0 => f(1) = S
    # But wait: the evaluator does f(1) = f(1) - S, then checks sum f(k)/k = 0.
    # Let's re-read: "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values"
    # So after replacement, new_f(1) = old_f(1) - S
    # Then sum new_f(k)/k = old_f(1)/1 - S + sum_{k>1} f(k)/k = (old_f(1) + sum_{k>1} f(k)/k) - S + sum_{k>1} f(k)/k
    # Wait, S = sum_k f(k)/k = f(1) + sum_{k>1} f(k)/k
    # So new sum = (f(1) - S)/1 + sum_{k>1} f(k)/k = f(1) - S + sum_{k>1} f(k)/k
    # = f(1) - (f(1) + sum_{k>1} f(k)/k) + sum_{k>1} f(k)/k = 0. Yes, it works.
    # So we just need to provide f(1) such that after the adjustment, everything is consistent.
    # Actually, the adjustment is done by the evaluator. We just provide the raw f.
    # The evaluator will handle the f(1) adjustment. So we don't need to do it ourselves.
    # But wait, the problem says "f(1) = 0 before this if 1 is not in the domain".
    # Since 1 is always square-free, mu(1) = 1, so 1 is in the domain.

    # Verify feasibility: M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, 10K]
    # For the exact Mobius function, M(x) = 1 exactly for all x >= 1.
    # So this should be feasible.

    # Compute the metric a = -sum_k f(k) log(k)/k
    # This is what the evaluator will compute (after the f(1) adjustment).
    # But we don't need to return it; the evaluator computes it.

    return domain, values


# EVOLVE-BLOCK-END
