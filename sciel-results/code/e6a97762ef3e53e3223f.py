# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    DEADLINE = 590.0  # safety margin before 630s kill
    
    # Compute Möbius function up to K using linear sieve
    K = 200000
    max_k = K
    
    mu = np.zeros(max_k + 1, dtype=np.int64)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(max_k + 1, dtype=bool)
    
    for i in range(2, max_k + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > max_k:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Only include non-zero mu values (where mu != 0)
    nonzero_mask = mu[1:] != 0
    domain = np.nonzero(nonzero_mask)[0] + 1
    values = mu[domain].astype(np.float64)
    
    m = len(domain)
    
    # Quick feasibility check on a few x values
    # For true Möbius, M(x) = sum_k mu(k)*floor(x/k) = 1 exactly for all x
    # Verify for x=1..50 and x=10K
    for x in list(range(1, 51)) + [10 * max_k]:
        idx = np.searchsorted(domain, x, side='right')
        if idx > 0:
            Mx = np.sum(values[:idx] * (x // domain[:idx]))
            if Mx > 1.0001:
                # Should not happen; fallback to smaller K
                K_small = 1000
                mu_small = np.zeros(K_small + 1, dtype=np.int64)
                mu_small[1] = 1
                primes_small = []
                is_comp_small = np.zeros(K_small + 1, dtype=bool)
                for i in range(2, K_small + 1):
                    if not is_comp_small[i]:
                        primes_small.append(i)
                        mu_small[i] = -1
                    for p in primes_small:
                        if i * p > K_small:
                            break
                        is_comp_small[i * p] = True
                        if i % p == 0:
                            mu_small[i * p] = 0
                            break
                        else:
                            mu_small[i * p] = -mu_small[i]
                nz = mu_small[1:] != 0
                domain = np.nonzero(nz)[0] + 1
                values = mu_small[domain].astype(np.float64)
                break
    
    elapsed = time.time() - start_time
    
    return domain, values

# EVOLVE-BLOCK-END
