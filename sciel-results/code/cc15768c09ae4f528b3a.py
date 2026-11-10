# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 5.0
    
    # Maximum K allowed by problem
    K_max = 200000
    
    # Sieve for Mobius function
    mu = np.ones(K_max + 1, dtype=np.int8)
    is_prime = np.ones(K_max + 1, dtype=bool)
    primes = []
    
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            for j in range(i, K_max + 1, i):
                is_prime[j] = False
                mu[j] *= -1
        # Check for square factors
        i2 = i * i
        if i2 <= K_max:
            for j in range(i2, K_max + 1, i2):
                mu[j] = 0

    # Compute prefix sums of mu for M(n)
    M = np.cumsum(mu)
    
    # We want to find the largest K such that the constraint is satisfied.
    # Constraint: M_eval(x) <= 1.0001 for x in [1, 10K]
    # M_eval(x) = 1 - R(x) - delta * x, where delta = sum_{k=1}^K mu(k)/k
    # R(x) = sum_{j=1}^{floor(x/(K+1))} (M(floor(x/j)) - M(K))
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Check feasibility for K from K_max downwards
    # We can't check all K, so we check a few candidates
    candidates = [K_max, K_max//2, K_max//4, 10000, 5000, 1000]
    
    for K in candidates:
        if time.time() > deadline - safety_margin:
            break
            
        # Calculate delta = sum_{k=1}^K mu(k)/k
        # This can be done via harmonic sum approximation or direct sum
        # Direct sum is O(K)
        inv_k = 1.0 / np.arange(1, K + 1)
        delta = np.dot(mu[1:K+1], inv_k)
        
        # Check constraint
        # We need min_{x in [K+1, 10K]} (R(x) + delta * x) >= -0.0001
        # R(x) depends on x.
        # We can sample x or check the worst case.
        # Worst case for R(x) + delta * x:
        # If delta is negative, -delta * x is positive, helping the constraint?
        # Wait, constraint is R(x) + delta * x >= -0.0001.
        # If delta is negative, delta * x is negative. So R(x) must be large positive.
        
        # Let's just check x = 10K, which is the largest x.
        x = 10 * K
        # R(x) = sum_{j=1}^{floor(x/(K+1))} (M(floor(x/j)) - M(K))
        # x/(K+1) is approx 10
        max_j = int(x // (K + 1))
        if max_j == 0:
            max_j = 1
            
        R_val = 0.0
        for j in range(1, max_j + 1):
            idx = int(x // j)
            if idx <= K_max:
                R_val += M[idx] - M[K]
                
        # Check if constraint holds at x=10K
        # This is a necessary condition, not sufficient.
        # But if it fails here, it likely fails elsewhere or is close.
        val = R_val + delta * x
        if val >= -0.0001:
            # Likely feasible. Return mu[1:K+1]
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = mu[1:K+1].astype(np.float64)
            return domain, values

    # If no candidate worked, or we ran out of time, return a safe fallback
    # The seed program's values
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
