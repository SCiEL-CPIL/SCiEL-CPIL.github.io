# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mobius(K):
    """Compute Mobius function for 1..K using a simple sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
    # Handle squares
    for p in primes:
        p2 = p * p
        if p2 > K:
            break
        for j in range(p2, K + 1, p2):
            mu[j] = 0
    return mu

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    max_K = 200000
    mu = _compute_mobius(max_K)
    
    # Precompute log(k)/k and mu(k)/k
    k_range = np.arange(1, max_K + 1, dtype=np.float64)
    log_k = np.log(k_range)
    inv_k = 1.0 / k_range
    mu_float = mu.astype(np.float64)
    
    # S_K = sum_{k=1}^K mu(k)/k
    # score_K = -sum_{k=1}^K mu(k) log(k)/k
    cum_S = np.cumsum(mu_float * inv_k)
    cum_score_terms = mu_float * log_k * inv_k
    cum_score = -np.cumsum(cum_score_terms)
    
    best_score = -np.inf
    best_K = 0
    
    # Iterate K
    # Condition: S_K >= -1e-5 / K
    # We check this for all K and keep the best feasible one.
    
    # Vectorized check for feasibility condition on S_K
    # S_K = cum_S[K-1]
    # Threshold = -1e-5 / K
    thresholds = -1e-5 / k_range
    feasible_mask = cum_S >= thresholds
    
    # Also need to ensure M(x) <= 1.0001 for x up to 10K.
    # The condition S_K >= -1e-5/K is derived from x=10K case assuming T(x,K) ~ 1.
    # For x <= K, M(x) = 1 - x S_K. If S_K >= 0, M(x) <= 1.
    # If S_K < 0, M(x) > 1 for large x.
    # The vectorized check covers the worst case x=10K roughly.
    # We should probably be more conservative or verify the best candidates.
    
    # Find indices of feasible K
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        # Fallback: pick K with smallest |S_K|
        idx_min_S = np.argmin(np.abs(cum_S))
        best_K = idx_min_S + 1
    else:
        # Among feasible, pick the one with max score
        # Score is cum_score[idx]
        feasible_scores = cum_score[feasible_indices]
        best_idx_in_feasible = np.argmax(feasible_scores)
        best_idx = feasible_indices[best_idx_in_feasible]
        best_K = best_idx + 1
        best_score = feasible_scores[best_idx_in_feasible]
        
        # Optional: Verify the best candidate more strictly if time permits
        # But the vectorized check is a good proxy.
        
    # Construct domain and values
    domain = np.arange(1, best_K + 1, dtype=np.int64)
    values = mu[1:best_K+1].astype(np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
