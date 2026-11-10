# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, leave margin
    
    # We suspect the Mobius function is the key.
    # We will test various K values to find the one that maximizes 'a' while satisfying M(x) <= 1.0001.
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Candidate K values. The seed is small. We try larger K.
    # The constraint M(x) <= 1.0001 is the bottleneck.
    # M(x) = sum_{d=1}^K mu(d) floor(x/d).
    # We need to check x up to 10K.
    
    # We'll generate mu up to a max K.
    max_K_to_try = 200000
    
    # Generate Mobius function up to max_K_to_try
    mu = np.zeros(max_K_to_try + 1, dtype=np.int64)
    mu[1] = 1
    primes = []
    is_prime = np.ones(max_K_to_try + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, max_K_to_try + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, max_K_to_try + 1, i):
                is_prime[j] = False
                mu[j] *= -1
                if j % (i*i) == 0:
                    mu[j] = 0
                    break
    
    # Function to evaluate a candidate K
    def evaluate_K(K):
        if K < 1:
            return -np.inf, None, None
        if time.time() > deadline - 60:
            return -np.inf, None, None
            
        # Domain is 1..K where mu is non-zero? Or all 1..K?
        # The task allows any domain. Including mu=0 doesn't help 'a' (log term 0? no, f=0 so term 0)
        # but adds to M(x) sum (0 contribution). So we can include all 1..K.
        # However, smaller domain is faster to check?
        # M(x) computation is O(K log K) or O(K * (10K/K)) = O(10K).
        # Actually M(x) for all x in 1..10K can be computed in O(K log K) using sieve-like update.
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:K + 1].astype(np.float64)
        
        # Check feasibility
        # M(x) = sum_{d=1}^K mu[d] * floor(x/d)
        # We need max_{x in 1..10K} M(x) <= 1.0001
        
        # Efficient computation of M(x) for x=1..10K
        # M(x) = sum_{d=1}^K mu[d] * (x // d)
        # This is a sum of step functions.
        # We can accumulate differences.
        # diff[i] += mu[d] for i in multiples of d? No.
        # M(x) is constant between integers? No, x is integer.
        # M(x) changes at every integer.
        
        # Let's use the property M(x) = 1 - T(x).
        # T(x) = sum_{d=K+1}^x mu[d] floor(x/d).
        # This is hard to compute fast for all x.
        
        # Direct computation for x=1..10K:
        # M = np.zeros(10*K + 1)
        # for d in 1..K:
        #   M[d::d] += mu[d]
        # This is O(K log K).
        
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        # Vectorized update might be hard for variable step sizes in simple loops.
        # We'll loop d.
        for d in range(1, K + 1):
            m_val = mu[d]
            if m_val != 0:
                # Add m_val to M[d], M[2d], ...
                # M[d::d] += m_val
                M[d::d] += m_val
        
        # Find max M(x)
        max_M = np.max(M[1:max_x+1])
        
        if max_M > 1.0001:
            return -np.inf, None, None
        
        # Calculate a
        # a = -sum f(k) log(k) / k
        # f(k) = mu(k)
        # sum mu(k) log(k) / k
        k_vals = domain.astype(np.float64)
        # Avoid log(0) or log(1) issues? log(1)=0.
        log_k = np.log(k_vals)
        a = -np.sum(values * log_k / k_vals)
        
        return a, domain, values

    # Try a set of K values
    # Start with small K to ensure we have a valid result, then try larger.
    # The seed K is 30.
    # Let's try K = 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000
    
    K_candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    
    # Sort by K ascending to find a valid one quickly?
    # Or descending to find the best 'a'?
    # 'a' is not monotonic.
    # But larger K generally closer to limit 1.
    # If seed is 1.0, and limit is 1.0, we might not beat it.
    # But if seed is small, we beat it easily.
    
    # Let's evaluate all candidates within time limit.
    results = []
    for K in K_candidates:
        if time.time() > deadline - 10:
            break
        score, dom, vals = evaluate_K(K)
        if score > -np.inf:
            results.append((score, K, dom, vals))
            if score > best_score:
                best_score = score
                best_domain = dom
                best_values = vals
    
    if best_domain is None:
        # Fallback to seed-like result
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    return best_domain, best_values

# EVOLVE-BLOCK-END
