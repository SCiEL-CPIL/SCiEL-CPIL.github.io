# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    # Max K to try
    K_max = 100000
    
    # Precompute Mobius function up to K_max
    # mu[1] = 1
    # mu[n] = 0 if n has a squared prime factor
    # mu[n] = (-1)^k if n is product of k distinct primes
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_prime = np.ones(K_max + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K_max + 1, i):
                is_prime[j] = False
                mu[j] *= -1
                if j % (i * i) == 0:
                    mu[j] = 0
        # Sieve logic for mu is slightly different, standard linear sieve is better
    
    # Linear Sieve for Mobius
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K_max + 1, dtype=bool)
    for i in range(2, K_max + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Precompute log(k)/k for score calculation
    ks = np.arange(1, K_max + 1, dtype=np.float64)
    log_k_over_k = np.log(ks) / ks
    log_k_over_k[0] = 0 # index 0 unused
    
    # Search for largest feasible K
    best_domain = None
    best_values = None
    best_score = 0.0
    
    # We check K from large to small
    # Step size to avoid too many checks
    step = 1000
    for K in range(K_max, 1000, -step):
        if time.time() > deadline - safety_margin:
            break
            
        # Compute f(1)
        # f(1) = - sum_{k=2}^K mu(k)/k
        # mu is 1-indexed in our array? 
        # mu[1] is at index 1.
        # sum is over k=2..K
        sum_mu_over_k = np.sum(mu[2:K+1].astype(np.float64) / ks[2:K+1])
        f1 = -sum_mu_over_k
        
        # Check feasibility
        # M(x) = f1 * x + sum_{k=2}^K mu(k) * floor(x/k)
        # We need max M(x) <= 1.0001 for x in 1..10K
        
        # Optimization: If f1 > 1.0001, M(1) = f1 > 1.0001, infeasible.
        if f1 > 1.0001:
            continue
            
        # If f1 is close to 1, we need to check the rest.
        # For speed, we can sample x or use a bound.
        # But let's compute exactly for a subset or all if K is small enough.
        
        # Estimate time for full check: 10*K^2 ops.
        # If K=10000, 10^9 ops.
        # We will compute M(x) for all x in 1..10K.
        
        x_max = 10 * K
        x = np.arange(1, x_max + 1, dtype=np.float64)
        M = f1 * x
        
        # Add contributions from k=2..K
        # Loop over k
        # To speed up, we can vectorize the floor division?
        # M += mu[k] * np.floor(x / k)
        
        # We can break early if M exceeds 1.0001?
        # No, M(x) is not monotonic.
        
        # Use a loop over k
        # We can optimize by only iterating k up to x_max, but K <= x_max/10.
        
        # To save time, we can check feasibility on a subset of x first?
        # The maximum of M(x) usually occurs at small x or where the tail error is large.
        # However, for safety, let's compute all.
        
        # If K is large, this loop is slow.
        # Let's limit K based on time.
        # 10 * K^2 < 5e8 (approx 1s) => K < 7000.
        # So we should only do full check for K < 7000?
        # Or use a faster method?
        
        # Alternative: Use the fact that M(x) = 1 + Error(x).
        # Error(x) = sum_{k=K+1}^x mu(k) floor(x/k)
        # This is also hard to compute.
        
        # Let's stick to direct computation but limit K.
        if K > 5000:
             # Use a heuristic check or smaller x_max?
             # The constraint is for x in [1, 10K].
             # If we check x in [1, 1000] and it's safe, is it safe for all?
             # Not necessarily.
             # But let's try to be smart.
             # We'll just skip large K if we don't have time.
             pass

        # For now, let's just try K up to 5000 with full check.
        if K <= 5000:
            for k in range(2, K + 1):
                if time.time() > deadline - safety_margin:
                    break
                m_val = mu[k]
                if m_val != 0:
                    M += m_val * np.floor(x / k)
            
            if np.max(M) > 1.0001:
                continue
        else:
            # For K > 5000, we assume it's feasible if f1 <= 1.0
            # This is a heuristic.
            if f1 > 1.0:
                continue
            # Accept it optimistically?
            # Risky. Let's just stop searching for large K.
            continue

        # Calculate score
        # a = - sum_{k=2}^K mu(k) log(k)/k
        score = -np.sum(mu[2:K+1].astype(np.float64) * log_k_over_k[2:K+1])
        
        if score > best_score:
            best_score = score
            # Construct domain and values
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = np.zeros(K, dtype=np.float64)
            values[0] = f1 # f(1)
            values[1:] = mu[2:K+1].astype(np.float64)
            best_domain = domain
            best_values = values

    # If no K found, use the seed
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END
