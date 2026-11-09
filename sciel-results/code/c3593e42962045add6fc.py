# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function scaled optimally to maximize the metric while satisfying
    the constraint M(x) <= 1.0001.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s safety margin
    
    # Compute Möbius function up to K
    K = 200000
    mu = _compute_mobius(K)
    
    # Domain: indices where mu(k) != 0
    domain = np.where(mu[1:] != 0)[0] + 1
    # Base values: mu(k)
    base_values = mu[domain].astype(np.float64)
    
    # Calculate S0 = sum_{k=1}^K mu(k)/k
    # Note: domain only contains k where mu(k) != 0, but 1/k is 0 for mu(k)=0 anyway.
    # So we can sum over the full range 1..K or just domain.
    # Using domain is faster if sparse, but mu is sparse (~60% zeros).
    # Let's sum over 1..K for simplicity and accuracy, or just domain.
    # mu[0] is unused.
    inv_k = 1.0 / np.arange(1, K + 1)
    S0 = np.sum(mu[1:] * inv_k)
    
    # Calculate L = sum_{k=1}^K mu(k) log(k) / k
    log_k = np.log(np.arange(1, K + 1))
    L = np.sum(mu[1:] * log_k * inv_k)
    
    # Determine optimal scaling factor c
    # M_c(x) = c * (1 - S0 * x)
    # Constraint: M_c(x) <= 1.0001 for x in [1, 10*K]
    # If S0 >= 0, max is at x=1: c * (1 - S0) <= 1.0001
    # If S0 < 0, max is at x=10*K: c * (1 - 10*K*S0) <= 1.0001
    
    x_max = 10 * K
    if S0 >= 0:
        max_val_base = 1 - S0
    else:
        max_val_base = 1 - S0 * x_max
        
    if max_val_base <= 0:
        # This shouldn't happen for reasonable K, but handle it
        c = 0.0
    else:
        c = 1.0001 / max_val_base
        
    # Clip c to ensure values stay within [-10, 10] after scaling
    # max value is c * max(abs(mu)) = c * 1
    # min value is c * min(mu) = -c
    # So we need c <= 10
    c = min(c, 10.0)
    
    # Apply scaling
    values = base_values * c
    
    # The normalization step in the evaluator:
    # f(1) is replaced by f(1) - S where S = sum f(k)/k
    # Our construction already satisfies sum f(k)/k = 0?
    # Let's check.
    # Sum_{k in domain} f(k)/k = c * Sum_{k in domain} mu(k)/k = c * S0.
    # Wait, the evaluator computes S = sum f(k)/k over the returned values.
    # Then it sets f(1) = f(1) - S.
    # If we return values such that sum f(k)/k = 0, then S=0 and f(1) is unchanged.
    # Does our scaled mu satisfy this?
    # Sum_{k=1}^K c mu(k)/k = c S0.
    # This is NOT 0 unless S0=0.
    # So the evaluator will modify f(1).
    # Let's re-evaluate M(x) with the evaluator's modification.
    
    # Returned values:
    # f_ret(k) = c mu(k) for k > 1
    # f_ret(1) = c mu(1) = c
    # S_eval = sum_{k=1}^K f_ret(k)/k = c S0
    # f_final(1) = f_ret(1) - S_eval = c - c S0 = c(1 - S0)
    # f_final(k) = c mu(k) for k > 1
    # This matches our derivation of M_c(x) = c(1 - S0 x).
    # So the logic holds.
    
    return domain, values


# EVOLVE-BLOCK-END
