# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    max_K_search = 2000
    deadline = 600 - 10 # 590s
    
    # Compute Mobius function up to max_K_search using linear sieve
    mu = np.zeros(max_K_search + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(max_K_search + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    lp = np.zeros(max_K_search + 1, dtype=np.int64) # least prime
    
    for i in range(2, max_K_search + 1):
        if lp[i] == 0:
            primes.append(i)
            lp[i] = i
            mu[i] = -1
        for p in primes:
            if p > lp[i] or i * p > max_K_search:
                break
            lp[i * p] = p
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Precompute log(k)/k
    k_range = np.arange(1, max_K_search + 1)
    log_k_over_k = np.log(k_range) / k_range
    
    # Compute cumulative sum of mu(k) * log(k) / k
    # This sum determines the objective a (a = -sum)
    # We want to minimize this sum to maximize a
    terms = mu[1:] * log_k_over_k
    cum_sum = np.cumsum(terms)
    
    # Find candidate K's: those with the smallest cum_sum
    # We take the top 20 smallest values to ensure we find a feasible one
    # Note: cum_sum index i-1 corresponds to K=i
    
    # Get indices of the 20 smallest values in cum_sum
    # argsort returns indices of cum_sum
    # We want the smallest values, so we take the first 20
    # But we must ensure K >= 1
    min_indices = np.argsort(cum_sum)[:20]
    candidate_Ks = [idx + 1 for idx in min_indices]
    
    # Add some specific K's just in case (e.g., small ones)
    candidate_Ks.extend([1, 2, 5, 10, 50, 100, 500, 1000])
    # Remove duplicates and sort by score (cum_sum value)
    unique_candidates = list(set(candidate_Ks))
    unique_candidates.sort(key=lambda k: cum_sum[k-1] if k <= max_K_search else np.inf)
    
    def check_feasibility(K):
        if K > max_K_search:
            return False
        S_K = np.sum(mu[1:K+1] / np.arange(1, K+1))
        max_x = 10 * K
        x = np.arange(1, max_x + 1)
        
        # Compute M_raw(x) = sum_{k=1}^K mu(k) * floor(x/k)
        M = np.zeros(max_x, dtype=np.float64)
        # Loop over k
        # This is O(K * 10K) operations, but vectorized per k
        for k in range(1, K + 1):
            M += mu[k] * (x // k)
        
        # Apply shift: M_adj(x) = M_raw(x) - S_K * x
        M -= S_K * x
        
        # Check constraint
        return np.max(M) <= 1.0001

    for K in unique_candidates:
        if time.time() > deadline:
            break
        if check_feasibility(K):
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = mu[1:K+1].astype(np.float64)
            return domain, values
            
    # Fallback to seed if no feasible K found (should not happen)
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
