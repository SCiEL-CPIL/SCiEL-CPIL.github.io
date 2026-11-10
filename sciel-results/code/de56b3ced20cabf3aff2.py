# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function mu(k) which satisfies sum_{k|x} mu(k) = 1 for x>=1,
    making M(x) = sum_k mu(k)*floor(x/k) = 1 for all x >= 1 (perfectly feasible).
    The objective a = -sum_k mu(k)*log(k)/k approaches 1 as K -> infinity.
    """
    start_time = time.time()
    DEADLINE = 620.0  # safety margin before 630s

    def compute_mobius(K):
        """Compute Mobius function mu[1..K] using a linear sieve."""
        mu = np.ones(K + 1, dtype=np.float64)
        is_composite = np.zeros(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1.0
            for p in primes:
                if i * p > K:
                    break
                is_composite[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0.0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def check_feasibility_vectorized(domain, values, K):
        """
        Check M(x) <= 1.0001 for all x in [1, 10K] using vectorized floor computation.
        M(x) = sum_k f(k) * floor(x/k)
        """
        max_x = 10 * K
        x_vals = np.arange(1, max_x + 1, dtype=np.float64)
        
        M = np.zeros(max_x, dtype=np.float64)
        
        for k, f in zip(domain, values):
            if f == 0.0:
                continue
            # floor(x/k) for all x in [1, max_x]
            floor_vals = (x_vals // k).astype(np.float64)
            M += f * floor_vals
        
        return np.max(M)

    def compute_score(domain, values):
        """Compute the objective a = -sum_k f(k)*log(k)/k"""
        return -np.sum(values * np.log(domain.astype(np.float64)) / domain.astype(np.float64))

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try progressively larger K values with the Möbius function
    K_candidates = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    
    for K in K_candidates:
        if time.time() > DEADLINE - 30:
            break
        
        mu = compute_mobius(K)
        
        # Build domain and values from non-zero mu
        nonzero_mask = mu[1:K + 1] != 0.0
        domain = np.where(nonzero_mask)[0] + 1
        values = mu[1:K + 1][nonzero_mask].copy()
        
        m = len(domain)
        
        # Check feasibility using vectorized approach
        max_val = check_feasibility_vectorized(domain, values, K)
        
        if max_val <= 1.0001:
            # Compute objective
            a = compute_score(domain, values)
            
            if a > best_score:
                best_score = a
                best_domain = domain.copy()
                best_values = values.copy()
        
        if time.time() > DEADLINE - 30:
            break
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: use the original seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
