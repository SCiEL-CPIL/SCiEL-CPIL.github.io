# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    max_K = 200000
    
    # We will try to use the largest K possible.
    # The cost is O(K log K).
    # Let's start with max_K and go down if it's too slow or infeasible.
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try K = 200000
    # If it fails or is too slow, try smaller K.
    
    for K in [200000, 100000, 50000, 10000]:
        if time.time() - start_time > 500:
            break
            
        try:
            # 1. Compute Mobius function up to K
            mu = np.zeros(K + 1, dtype=np.int8)
            mu[1] = 1
            is_prime = np.ones(K + 1, dtype=bool)
            primes = []
            
            # Standard sieve for Mobius
            # mu[n] = 0 if n has a squared prime factor
            # mu[n] = (-1)^k if n is product of k distinct primes
            
            # Efficient way:
            # 1. Initialize mu[i] = 1
            # 2. For each prime p, for multiples of p, mu[m] *= -1
            # 3. For multiples of p^2, mu[m] = 0
            
            mu[:] = 1
            is_prime = np.ones(K + 1, dtype=bool)
            is_prime[0] = False
            is_prime[1] = False
            
            for i in range(2, int(np.sqrt(K)) + 1):
                if is_prime[i]:
                    for j in range(i*i, K + 1, i):
                        is_prime[j] = False
                    # Update mu for multiples of i
                    for j in range(i, K + 1, i):
                        mu[j] *= -1
                    # Update mu for multiples of i^2
                    i_sq = i * i
                    for j in range(i_sq, K + 1, i_sq):
                        mu[j] = 0
            
            # Note: The above loop for mu update is O(K log log K) roughly.
            # But the inner loops are Python loops, which might be slow for K=200k.
            # Let's use a faster sieve implementation if possible, or accept the time.
            # 200k is small enough for Python loops?
            # Sum K/p ~ K log log K. 200k * 3 ~ 600k ops. Very fast.
            
            # 2. Compute S = sum(mu[k]/k)
            # k=1 to K
            # mu[0] is 0, so we can ignore index 0
            k_vals = np.arange(1, K + 1)
            mu_vals = mu[1:K+1].astype(np.float64)
            S = np.sum(mu_vals / k_vals)
            
            # 3. Compute M(x) for x in [1, 10K]
            # M(x) = sum_{k=1}^K mu[k] * floor(x/k)
            # We compute A[n] = sum_{k|n, k<=K} mu[k]
            # Then M(x) = sum_{n=1}^x A[n]
            
            max_x = 10 * K
            A = np.zeros(max_x + 1, dtype=np.float64)
            
            # Iterate k from 1 to K
            # If mu[k] != 0, add mu[k] to A[multiples of k]
            # To speed up, we can use slicing if we vectorize, but loop is fine for K=200k?
            # 200k iterations. Inside, we update A.
            # Total updates ~ 10K log K.
            
            # Optimization: Only iterate non-zero mu?
            # mu[k] is 0 for many k.
            # But finding non-zeros requires a list.
            
            # Let's just loop.
            for k in range(1, K + 1):
                v = mu[k]
                if v != 0:
                    # Add v to A[k], A[2k], ...
                    # A[k::k] += v
                    # This is a NumPy slice operation, very fast.
                    A[k::k] += v
            
            # Compute cumulative sum to get M(x)
            M = np.cumsum(A)
            
            # 4. Check Constraint
            # M'(x) = M(x) - S * x
            # x ranges from 1 to 10K
            x_vals = np.arange(1, max_x + 1)
            M_prime = M[1:max_x+1] - S * x_vals
            
            max_M_prime = np.max(M_prime)
            
            if max_M_prime <= 1.0001:
                # Feasible
                # Calculate Score a = - sum(mu[k] * log(k) / k)
                log_k = np.log(k_vals)
                a = -np.sum(mu_vals * log_k / k_vals)
                
                # Prepare output
                domain = k_vals.astype(np.int64)
                values = mu_vals.copy()
                
                # We want to maximize a.
                # Since a increases with K (converging to 1), the largest feasible K is best.
                # So we can return immediately if K=200000 is feasible.
                # If not, we continue to smaller K.
                
                if K == 200000:
                    return domain, values
                
                # If we are here, K < 200000 was tried? No, we start with 200000.
                # If 200000 fails, we try 100000, etc.
                # So we just store the best feasible one found so far.
                # But since we iterate from large to small, the first feasible one is the best.
                return domain, values
                
        except Exception as e:
            # If an error occurs, try smaller K
            continue
            
    # If no K worked (unlikely), return the seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
