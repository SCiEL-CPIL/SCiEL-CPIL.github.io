# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_mobius(n):
    """Compute Möbius function mu(k) for k = 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def check_feasibility_fast(mu, K, max_x):
    """
    Check if M(x) <= 1.0001 for all x in [1, max_x].
    Uses divisor-sum approach: M(x) = sum_{n=1}^{x} g(n) where g(n) = sum_{k|n, k<=K} mu(k).
    Compute g(n) via sieve, then prefix sum.
    """
    # g(n) = sum_{k|n, k<=K} mu(k)
    # For each k with mu(k) != 0, add mu(k) to all multiples of k up to max_x
    g = np.zeros(max_x + 1, dtype=np.float64)
    
    # Only process k where mu(k) != 0
    for k in range(1, K + 1):
        mk = mu[k]
        if mk == 0:
            continue
        # Add mk to g[j] for all j that are multiples of k
        g[k::k] += mk
    
    # M(x) = sum_{n=1}^{x} g(n) = prefix sum of g
    M = np.cumsum(g)
    
    M_max = np.max(M)
    return M_max <= 1.0001, M_max

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s limit
    
    max_K = 200000
    mu = compute_mobius(max_K)
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try K values from large to small
    K_list = [200000, 150000, 100000, 75000, 50000, 30000, 20000, 10000, 5000]
    
    for K in K_list:
        if time.time() > deadline - 30:
            break
        
        max_x = 10 * K
        
        # Check feasibility using fast divisor-sum method
        feasible, M_max = check_feasibility_fast(mu, K, max_x)
        
        if feasible:
            # Extract domain and values where mu(k) != 0
            mask = mu[1:K + 1] != 0
            domain = np.nonzero(mask)[0] + 1
            values = mu[1:K + 1][mask].astype(np.float64)
            
            # Compute the score a = -sum f(k) * log(k) / k
            log_k = np.log(domain.astype(np.float64))
            a = -np.sum(values * log_k / domain.astype(np.float64))
            
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
                
            # Found feasible solution with large K, stop
            if K >= 50000:
                break
    
    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
