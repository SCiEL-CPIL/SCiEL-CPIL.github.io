# EVOLVE-BLOCK-START
import numpy as np
import time

def _sieve_mobius(K):
    """Compute Mobius function mu[1..K] using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu, primes

def prime_number_theorem_partial_function():
    """
    Construct a partial function f to maximize the objective.
    
    Strategy:
    1. Use the Mobius function mu(k) as the base solution.
    2. Scale mu(k) by a factor s to satisfy M(x) <= 1.0001.
    3. Determine the optimal s by finding the minimum of (1.0001 / (M_full(x) - x*S_K)) 
       over x where the denominator is positive.
    """
    start_time = time.time()
    deadline = 590.0
    
    # Try large K first
    for K in [200000, 100000, 50000, 20000, 10000]:
        if time.time() > deadline - 30:
            break
        
        # Compute Mobius function
        mu, primes = _sieve_mobius(K)
        
        # Compute S_K = sum_{k=1}^K mu(k)/k
        ks = np.arange(1, K + 1, dtype=np.float64)
        S_K = np.sum(mu[1:K+1].astype(np.float64) / ks)
        
        # Compute M_full(x) = sum_{k=1}^K mu(k) floor(x/k) for x in [1, 10K]
        max_x = 10 * K
        M_full = np.zeros(max_x + 1, dtype=np.float64)
        for d in range(1, K + 1):
            if mu[d] != 0:
                M_full[d::d] += mu[d]
        
        # Compute D(x) = M_full(x) - x * S_K
        x_vals = np.arange(1, max_x + 1, dtype=np.float64)
        D = M_full[1:max_x+1] - x_vals * S_K
        
        # Find s_max such that s * D(x) <= 1.0001 for all x
        # s <= 1.0001 / D(x) for D(x) > 0
        positive_D = D > 0
        if np.any(positive_D):
            s_max_constraint = np.min(1.0001 / D[positive_D])
        else:
            s_max_constraint = 10.0 # Large value
        
        # Constraint from bounds: |f(k)| <= 10
        # f(1) = s(1 - S_K), f(k) = s mu(k)
        # |f(1)| <= 10 => s <= 10 / |1 - S_K|
        # |f(k)| <= 10 => s <= 10 / max(|mu(k)|) = 10
        s_bound = 10.0 / max(1.0, abs(1.0 - S_K))
        
        s = min(s_max_constraint, s_bound)
        
        # If s is very small, try smaller K? Or just use it.
        # Check if s is reasonable
        if s < 0.1:
            continue
        
        # Construct f
        f = np.empty(K + 1, dtype=np.float64)
        f[0] = 0.0
        f[1] = s * (1.0 - S_K)
        f[2:K+1] = s * mu[2:K+1].astype(np.float64)
        
        # Verify feasibility
        A = np.zeros(max_x + 1, dtype=np.float64)
        for d in range(1, K + 1):
            if f[d] != 0:
                A[d::d] += f[d]
        M_check = np.cumsum(A)
        
        if np.max(M_check) <= 1.0001:
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = f[1:K + 1].copy()
            return domain, values
    
    # Fallback: use the known good small solution
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
