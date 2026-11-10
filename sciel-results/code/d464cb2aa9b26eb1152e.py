# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
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

    def check_feasibility_fast(domain, values, K):
        """
        Check M(x) <= 1.0001 for all x in [1, 10K] using efficient range updates.
        Returns the maximum value of M(x).
        """
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        for k, f in zip(domain, values):
            if f == 0.0:
                continue
            # For each j, add f to M[j*k : (j+1)*k]
            # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k, max_x + 1)
                M[lo:hi] += f
                j += 1
        
        return np.max(M[1:max_x + 1])

    def compute_objective(domain, values):
        """Compute a = -sum f(k)*log(k)/k"""
        # Note: log(1) = 0, so f(1) doesn't contribute to the sum
        return -np.sum(values * np.log(domain) / domain)

    # Try progressively larger K values
    K_candidates = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    for K in K_candidates:
        if time.time() > DEADLINE - 120:
            break
        
        mu = compute_mobius(K)
        
        # Build domain and values from non-zero mu
        nonzero_mask = mu[1:K + 1] != 0.0
        domain_base = np.where(nonzero_mask)[0] + 1
        values_base = mu[1:K + 1][nonzero_mask].copy()
        
        # Ensure 1 is in domain (mu[1] = 1)
        if len(domain_base) == 0 or domain_base[0] != 1:
            domain_base = np.concatenate([[1], domain_base])
            values_base = np.concatenate([[1.0], values_base])
        
        m = len(domain_base)
        
        # Check feasibility
        max_val = check_feasibility_fast(domain_base, values_base, K)
        
        if max_val <= 1.0001:
            # Compute objective
            a = compute_objective(domain_base, values_base)
            
            if a > best_score:
                best_score = a
                best_domain = domain_base.copy()
                best_values = values_base.copy()
        
        if time.time() > DEADLINE - 120:
            break
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: use the original seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
