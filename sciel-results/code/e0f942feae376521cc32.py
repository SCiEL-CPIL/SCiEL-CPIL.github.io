# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    DEADLINE = 610.0  # safety margin before 630s

    def check_feasibility_fast(domain, values, K):
        """Check M(x) <= 1.0001 for all x in [1, 10K] using efficient range updates."""
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        for k, f in zip(domain, values):
            if f == 0.0:
                continue
            # For each j, add f to M[j*k : (j+1)*k]
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k, max_x + 1)
                M[lo:hi] += f
                j += 1
        
        return np.max(M[1:max_x + 1])

    def compute_score(domain, values):
        """Compute the objective a = -sum_k f(k)*log(k)/k"""
        return -np.sum(values * np.log(domain) / domain)

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Strategy 1: f(k) = -c/k for k in S, f(1) = c*sum_{k in S} 1/k^2
    # a = c*sum_{k in S} log(k)/k^2
    # Try different sets S and find optimal c

    # Try S = {2, 3, ..., K} for various K
    for K in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000]:
        if time.time() > DEADLINE - 60:
            break
        
        ks = np.arange(2, K + 1, dtype=np.float64)
        
        # f(k) = -c/k for k >= 2, f(1) = c * sum(1/k^2)
        # sum f(k)/k = -c*sum(1/k^2) + f(1) = 0 => f(1) = c*sum(1/k^2)
        sum_inv_sq = np.sum(1.0 / ks**2)
        
        # a = -sum f(k)*log(k)/k = -[(-c*sum log(k)/k^2) + 0] = c*sum log(k)/k^2
        a_coeff = np.sum(np.log(ks) / ks**2)
        
        # M(x) = f(1)*floor(x/1) + sum_{k=2}^K (-c/k)*floor(x/k)
        #      = c*sum_inv_sq*x - c*sum_{k=2}^K floor(x/k)/k
        #      = c * [sum_inv_sq*x - sum_{k=2}^K floor(x/k)/k]
        
        # We need M(x) <= 1.0001 for all x in [1, 10K]
        # M(x) = c * g(x) where g(x) = sum_inv_sq*x - sum_{k=2}^K floor(x/k)/k
        # Need c * max(g(x)) <= 1.0001, so c <= 1.0001 / max(g(x))
        
        # Compute g(x) for x in [1, 10K]
        max_x = 10 * K
        g = np.zeros(max_x + 1, dtype=np.float64)
        
        # g(x) = sum_inv_sq * x - sum_{k=2}^K floor(x/k)/k
        # sum_{k=2}^K floor(x/k)/k can be computed efficiently
        # For each k, floor(x/k)/k is added to g(x) for x in [k, 2k-1], [2k, 3k-1], etc.
        
        # Actually, let's just compute it directly for small K
        if K <= 10000:
            for x in range(1, max_x + 1):
                s = 0.0
                for k in range(2, K + 1):
                    if k > x:
                        break
                    s += (x // k) / k
                g[x] = sum_inv_sq * x - s
        
        # Find max g(x)
        max_g = np.max(g[1:max_x + 1])
        
        if max_g > 0:
            c_max = 1.0001 / max_g
            c = c_max  # Use maximum feasible c to maximize a
            
            # Build domain and values
            domain = np.concatenate([[1], np.arange(2, K + 1, dtype=np.int64)])
            values = np.empty(K, dtype=np.float64)
            values[0] = c * sum_inv_sq  # f(1)
            values[1:] = -c / np.arange(2, K + 1, dtype=np.float64)
            
            # Verify feasibility
            max_val = check_feasibility_fast(domain, values, K)
            
            if max_val <= 1.0001:
                a = compute_score(domain, values)
                
                if a > best_score:
                    best_score = a
                    best_domain = domain.copy()
                    best_values = values.copy()
        
        if time.time() > DEADLINE - 60:
            break
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: use the original seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
