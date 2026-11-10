# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes the objective
    a = -sum_k f(k)*log(k)/k subject to the constraint that
    M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K].

    The ideal construction is f(k) = mu(k) (the Mobius function), for which
    sum_{k<=x} mu(k)*floor(x/k) = 1 exactly for all x. The objective a
    converges to 1.0 as K -> infinity.

    We compute mu(k) up to K using a sieve, then check feasibility and
    return the largest feasible K (up to 200000).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    # Compute Mobius function up to max K
    MAX_K = 200000
    
    # Sieve for Mobius function
    mu = np.ones(MAX_K + 1, dtype=np.int64)
    primes = []
    is_prime = np.ones(MAX_K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, MAX_K + 1):
        if is_prime[i]:
            primes.append(i)
            # Mark multiples of i
            for j in range(i, MAX_K + 1, i):
                is_prime[j] = False if j != i else True
                mu[j] *= -1
            i2 = i * i
            if i2 <= MAX_K:
                for j in range(i2, MAX_K + 1, i2):
                    mu[j] = 0
    
    # Re-sieve properly: standard Mobius sieve
    mu = np.ones(MAX_K + 1, dtype=np.int64)
    is_prime = np.ones(MAX_K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
    for i in range(2, MAX_K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, MAX_K + 1, i):
                is_prime[j] = False if j > i else True
                mu[j] *= -1
            sq = i * i
            if sq <= MAX_K:
                for j in range(sq, MAX_K + 1, sq):
                    mu[j] = 0
    
    # Fix: the above sieve logic is flawed. Use a cleaner approach.
    # Standard linear sieve for Mobius
    mu = np.zeros(MAX_K + 1, dtype=np.int64)
    mu[1] = 1
    is_composite = np.zeros(MAX_K + 1, dtype=bool)
    primes = []
    
    for i in range(2, MAX_K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > MAX_K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Now mu[0..MAX_K] is computed. mu[0]=0, mu[1]=1, etc.
    
    # The domain will be indices where mu != 0 (squarefree numbers), plus index 1
    # Actually, we include all k from 1 to K where mu(k) != 0
    # But for the constraint check, we need all k (mu=0 contributes nothing to M(x))
    
    # Try K from MAX_K downward
    best_K = 0
    best_domain = None
    best_values = None
    best_obj = -np.inf
    
    # We'll try a few candidate K values
    candidates = [MAX_K]
    # Add some spaced-out candidates in case large K fails
    for k in [150000, 100000, 75000, 50000, 25000, 10000, 5000, 2000, 1000]:
        candidates.append(k)
    
    for K in candidates:
        if time.time() > deadline:
            break
        if K <= best_K:
            continue
        
        # Build domain and values for this K
        # Include only k where mu(k) != 0 to keep arrays small
        mask = (np.arange(1, K + 1) >= 1) & (mu[1:K+1] != 0)
        k_idx = np.where(mask)[0] + 1  # actual k values (1-indexed)
        mu_vals = mu[1:K+1][mask]
        
        if len(k_idx) == 0:
            continue
        
        # Clip values to [-10, 10] (mu is already in {-1, 0, 1}, so no effect)
        vals = np.clip(mu_vals.astype(np.float64), -10.0, 10.0)
        
        # Compute S = sum_k f(k)/k over clipped values
        S = np.sum(vals / k_idx.astype(np.float64))
        
        # Replace f(1) by f(1) - S
        # Find index where k_idx == 1
        idx1 = np.where(k_idx == 1)[0]
        if len(idx1) > 0:
            vals[idx1[0]] -= S
        # If 1 is not in domain, it stays 0 (no contribution)
        
        # Verify sum_k f(k)/k is now ~ 0
        S_check = np.sum(vals / k_idx.astype(np.float64))
        # It should be 0 (up to floating point)
        
        # Compute M(x) for x in [1, 10K]
        # M(x) = sum_k f(k) * floor(x/k)
        # Efficient computation: for each k, add f(k) to M(x) for x in [k, 10K]
        # in steps of k
        limit_x = 10 * K
        M = np.zeros(limit_x + 1, dtype=np.float64)
        
        for k_val, v_val in zip(k_idx, vals):
            if v_val == 0.0:
                continue
            # Add v_val to M[k_val : limit_x+1 : k_val]
            M[k_val:limit_x+1:k_val] += v_val
        
        # Check constraint: max(M[1:]) <= 1.0001
        max_M = np.max(M[1:])
        
        if max_M <= 1.0001:
            # Feasible!
            # Compute objective a = -sum_k f(k) * log(k) / k
            log_k = np.log(k_idx.astype(np.float64))
            obj = -np.sum(vals * log_k / k_idx.astype(np.float64))
            
            if obj > best_obj:
                best_obj = obj
                best_K = K
                best_domain = k_idx
                best_values = vals
        
        if best_K == MAX_K:
            break
    
    if best_domain is None:
        # Fallback: tiny feasible function
        domain = np.array([1, 2, 3], dtype=np.int64)
        values = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        # Check feasibility
        S = np.sum(values / domain.astype(np.float64))
        idx1 = np.where(domain == 1)[0]
        if len(idx1) > 0:
            values[idx1[0]] -= S
        limit_x = 30
        M = np.zeros(limit_x + 1, dtype=np.float64)
        for k_val, v_val in zip(domain, values):
            if v_val != 0:
                M[k_val:limit_x+1:k_val] += v_val
        if np.max(M[1:]) > 1.0001:
            # Just return something trivially safe
            domain = np.array([1], dtype=np.int64)
            values = np.array([0.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values
    
    return domain, values


# EVOLVE-BLOCK-END
