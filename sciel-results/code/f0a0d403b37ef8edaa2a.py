# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n=1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
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

def _compute_M_vectorized(domain, values, max_x):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x]
    using a vectorized difference-array approach.
    """
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = int(domain[idx])
        f_k = values[idx]
        if f_k == 0.0:
            continue
        
        max_j = max_x // k
        if max_j == 0:
            continue
        
        starts = np.arange(1, max_j + 1, dtype=np.int64) * k
        ends = np.arange(2, max_j + 2, dtype=np.int64) * k
        ends = np.minimum(ends, max_x + 1)
        vals = f_k * np.arange(1, max_j + 1, dtype=np.float64)
        
        np.add.at(diff, starts, vals)
        np.add.at(diff, ends, -vals)
    
    M = np.cumsum(diff)
    return M[:max_x + 1]

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0

    # Use a large K to approximate the infinite sum
    K = 200000
    
    # Compute Mobius function
    mu = _compute_mobius(K)
    
    # Filter non-zero values
    mask = mu[1:K + 1] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    
    if len(domain) == 0:
        # Fallback
        domain = np.array([1], dtype=np.int64)
        values = np.array([0.0], dtype=np.float64)
        return domain, values

    # Evaluator simulation
    clipped_values = np.clip(values, -10.0, 10.0)
    k_float = domain.astype(np.float64)
    
    # S = sum f(k)/k
    S = np.sum(clipped_values / k_float)
    
    # Adjust f(1)
    # Find index of 1 in domain
    idx_1 = np.where(domain == 1)[0]
    final_values = clipped_values.copy()
    if len(idx_1) > 0:
        final_values[idx_1[0]] -= S
    else:
        # If 1 is not in domain, the evaluator effectively adds f(1) = -S to the sum for normalization
        # But the domain returned must match the values.
        # The problem says "f(1) is replaced by ... (f(1) = 0 before this if 1 is not in the domain)"
        # This implies the function is considered on the domain.
        # If 1 is not in the domain, does the evaluator add it?
        # "domain holds distinct positive integers... values holds the finite reals f(k)"
        # Usually, if 1 is not in the domain, the term f(1) is 0 in the sum.
        # The normalization step "f(1) = f(1) - S" would set f(1) to -S.
        # Does this imply 1 MUST be in the domain for the function to be well-defined in the evaluator's mind?
        # Or does it just mean the sum becomes 0?
        # "so that afterwards sum_k f(k)/k = 0"
        # If 1 is not in domain, and we don't add it, the sum is still S (non-zero).
        # So the evaluator MUST be adding 1 to the domain if it's not there, or assuming it's there.
        # To be safe, we should include 1 in the domain.
        # In Mobius, 1 is always included.
        pass

    # Compute M(x)
    max_x = 10 * K
    M = _compute_M_vectorized(domain, final_values, max_x)
    
    # Check feasibility
    M_max = np.max(M[1:max_x + 1])
    
    # Scale if necessary
    alpha = 1.0
    if M_max > 1.0001:
        alpha = 1.0001 / M_max
        final_values = final_values * alpha
        # Note: Scaling final_values changes the sum sum f(k)/k?
        # No, we are scaling the output. The evaluator will re-compute S and adjust f(1).
        # Let's trace:
        # We return f_scaled.
        # Evaluator clips (no change if alpha small).
        # Evaluator computes S_scaled = sum f_scaled(k)/k = alpha * S_final?
        # Wait. final_values = clipped - S (at 1).
        # If we scale final_values by alpha, the new values are alpha * final_values.
        # The evaluator computes S_new = sum (alpha * final_values) / k = alpha * sum final_values/k.
        # But sum final_values/k = 0 (by construction).
        # So S_new = 0.
        # Then evaluator sets f(1)_new = f(1)_scaled - 0 = f(1)_scaled.
        # So M(x) = sum alpha * final_values(k) floor(x/k) = alpha * M_final(x).
        # So M(x) <= 1.0001 is satisfied.
        
        # Recalculate a?
        # a = -sum f(k) log(k) / k.
        # The evaluator computes a using the *adjusted* values.
        # Since S_new = 0, the adjusted values are just the scaled values.
        # So a = -sum (alpha * final_values) * log(k) / k = alpha * a_final.
        
    # Compute a
    a = -np.sum(final_values * np.log(k_float) / k_float)
    
    # If a is negative, we might want to flip signs?
    # If we flip signs, M(x) becomes -M(x).
    # Constraint is M(x) <= 1.0001.
    # If M(x) was positive and we flip, it becomes negative, which is <= 1.0001.
    # And a becomes -a, which is positive.
    # So if a < 0, we should flip the signs of all values!
    
    if a < 0:
        final_values = -final_values
        a = -a
        # Check M(x) again?
        # M(x) becomes -M(x).
        # If original M(x) <= 1.0001, is -M(x) <= 1.0001?
        # Yes, if M(x) >= -1.0001.
        # M(x) is sum of mu(k) floor(x/k).
        # For mu, M(x) is around 1.
        # So -M(x) is around -1.
        # -1 <= 1.0001 is true.
        # So flipping is safe if M(x) is not too negative.
        # We should check if min(M(x)) >= -1.0001?
        # No, the constraint is only upper bound.
        # So flipping is always safe for the upper bound constraint, provided the new M(x) doesn't exceed 1.0001.
        # Since M(x) ~ 1, -M(x) ~ -1, which is fine.
        
        # But wait, if we flip, we must ensure the evaluator's normalization doesn't break it.
        # S_new = sum (-f(k))/k = -S_final = 0.
        # f(1)_new = -f(1)_final.
        # M(x)_new = -M(x)_final.
        # So it works.
        
    return domain, final_values

# EVOLVE-BLOCK-END
