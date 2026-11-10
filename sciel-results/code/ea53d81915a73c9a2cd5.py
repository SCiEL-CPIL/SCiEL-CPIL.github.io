# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f approximating the Mobius function to maximize the objective.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Choose K. Larger K gives better approximation of 1.
    # K=50000 is safe and fast.
    K = 50000
    
    # 1. Compute Mobius function up to K
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    # Sieve of Eratosthenes to compute mu
    mu[0] = 0
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

    # 2. Check feasibility: M(x) = sum_{n<=x} A(n) where A(n) = sum_{d|n, d<=K} mu(d)
    # We need M(x) <= 1.0001 for all x in [1, 10K]
    
    max_x = 10 * K
    A = np.zeros(max_x + 1, dtype=np.float64)
    
    # Efficiently compute A(n)
    # For each d, add mu(d) to A[d], A[2d], ...
    for d in range(1, K + 1):
        if mu[d] != 0:
            # Add mu[d] to indices d, 2d, ... up to max_x
            A[d:max_x+1:d] += mu[d]
            
    # M(x) is cumulative sum of A
    M = np.cumsum(A)
    
    # Check constraint
    # M[1] should be 1. M[x] for x>1 should be close to 1.
    # The constraint is M(x) <= 1.0001
    if np.max(M) > 1.0001:
        # If infeasible, try a smaller K or fallback
        # For now, if it fails, we might need to scale down or adjust.
        # However, for Mobius, it is theoretically 1 for x<=K.
        # For x>K, errors accumulate.
        # If max > 1.0001, we can try to reduce the "tail" or just accept if it's close?
        # No, must be feasible.
        # Let's try to find the max violation.
        # If it's slightly over, we can't easily fix without changing f.
        # But for K=50000, the error is very small.
        # Let's assume it passes. If not, we return a smaller K.
        # We'll implement a fallback to K=10000 if this fails.
        K = 10000
        mu = np.ones(K + 1, dtype=np.float64)
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        mu[0] = 0
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
        
        max_x = 10 * K
        A = np.zeros(max_x + 1, dtype=np.float64)
        for d in range(1, K + 1):
            if mu[d] != 0:
                A[d:max_x+1:d] += mu[d]
        M = np.cumsum(A)
        
        if np.max(M) > 1.0001:
            # Fallback to the seed program's known good values if all else fails
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values

    # 3. Construct output
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K + 1].copy()
    
    # Optional: Clip values to [-10, 10] as per evaluator? 
    # The evaluator clips values to [-10, 10]. Mu is -1, 0, 1. So no change.
    
    # Time check
    if time.time() > deadline - 10:
        # Return best so far (which is this)
        pass

    return domain, values

# EVOLVE-BLOCK-END
