# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n = 1..K using sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    mu[1] = 1
    return mu


def _check_feasibility(domain, values, K):
    """Check if M(x) <= 1.0001 for all x in [1, 10*K].
    M(x) = sum_k f(k) * floor(x/k)
    """
    m = len(domain)
    if m == 0:
        return True, 0.0
    
    max_x = 10 * K
    
    # h(n) = sum_{k in domain, k | n} f(k)
    # M(x) = sum_{n=1}^{x} h(n)
    
    h = np.zeros(max_x + 1, dtype=np.float64)
    
    # For each k in domain, add f(k) to h(n) for all n that are multiples of k
    for i in range(m):
        k = domain[i]
        v = values[i]
        if v == 0.0:
            continue
        # Add v to h at indices k, 2k, 3k, ... up to max_x
        h[k::k] += v
    
    # M(x) = cumsum(h)
    M_prefix = np.cumsum(h)
    max_M = np.max(M_prefix)
    
    return max_M <= 1.0001, max_M


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 570  # Leave 30s safety margin
    
    # Compute Möbius function up to K = 200000
    K = 200000
    mu = _compute_mobius(K)
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    values = np.clip(values, -10, 10)
    
    # Compute S = sum_k f(k)/k
    S = np.sum(values / domain.astype(np.float64))
    
    # Adjust f(1): f(1) -= S
    values[0] -= S  # domain[0] = 1
    
    # Check feasibility
    feasible, max_M = _check_feasibility(domain, values, K)
    
    if not feasible:
        # Try smaller K values
        for K_try in [100000, 50000, 30000, 20000, 10000, 5000, 2000, 1000, 500]:
            if time.time() > deadline:
                break
            
            mu_try = _compute_mobius(K_try)
            domain_try = np.arange(1, K_try + 1, dtype=np.int64)
            values_try = mu_try[1:].astype(np.float64)
            values_try = np.clip(values_try, -10, 10)
            
            S_try = np.sum(values_try / domain_try.astype(np.float64))
            values_try[0] -= S_try
            
            feasible_try, max_M_try = _check_feasibility(domain_try, values_try, K_try)
            
            if feasible_try:
                domain = domain_try
                values = values_try
                break
    
    return domain, values


# EVOLVE-BLOCK-END
