# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    DEADLINE = 600.0
    
    K = 200000
    
    # Efficient Mobius sieve
    mu = np.ones(K + 1, dtype=np.float64)
    mu[0] = 0.0
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    
    mu[1] = 1.0
    
    # Build domain: all k where mu[k] != 0
    mask = mu != 0
    domain = np.where(mask)[0]
    mu_vals = mu[domain]
    
    # Compute sigma_K = sum mu(k)/k over domain
    sigma_K = np.sum(mu_vals / domain)
    
    x_max = 10 * K
    
    # Estimate max c based on M(x) approx c(1 - sigma_K * x)
    # The term (1 - sigma_K * x) can be negative if sigma_K is large positive and x large.
    # But sigma_K is small.
    # We need c * (1 - sigma_K * x) <= 1.0001
    # If 1 - sigma_K * x > 0, c <= 1.0001 / (1 - sigma_K * x)
    # If 1 - sigma_K * x <= 0, the constraint is automatically satisfied for c > 0?
    # No, M(x) must be <= 1.0001. If the term is negative, M(x) is negative (assuming c>0), which is <= 1.0001.
    # So we only care about the maximum of (1 - sigma_K * x) over x where it is positive.
    
    # Let's find the max value of (1 - sigma_K * x) for x in [1, x_max]
    # Since sigma_K is likely small, 1 - sigma_K * x is decreasing if sigma_K > 0, increasing if sigma_K < 0.
    if sigma_K >= 0:
        max_term = 1.0 - sigma_K * 1.0 # at x=1
        # Check if it becomes negative
        if 1.0 - sigma_K * x_max < 0:
             # It crosses zero. The max is at x=1.
             max_term = 1.0 - sigma_K
        else:
             max_term = 1.0 - sigma_K
    else:
        # sigma_K < 0, term increases with x
        max_term = 1.0 - sigma_K * x_max
        
    if max_term <= 0:
        # This implies M(x) is always negative? Unlikely.
        c_est = 10.0
    else:
        c_est = 1.0001 / max_term
    
    # Safety margin
    c = c_est * 0.9
    
    # Function to check feasibility
    def check_feasibility(c_val):
        values = mu_vals * c_val
        values = np.clip(values, -10.0, 10.0)
        
        # Apply f(1) adjustment
        S = np.sum(values / domain)
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            values[idx_1[0]] -= S
        
        # Check M(x) at critical points
        # x=1, x=K, x=10K
        check_xs = np.array([1, K, x_max], dtype=np.int64)
        for x in check_xs:
            floors = np.floor(x / domain).astype(np.int64)
            Mx = np.sum(values * floors)
            if Mx > 1.0001:
                return False, values
        return True, values

    feasible, values = check_feasibility(c)
    
    if not feasible:
        # Binary search for c
        c_low = 0.0
        c_high = c
        for _ in range(20):
            c_mid = (c_low + c_high) / 2.0
            feasible, values = check_feasibility(c_mid)
            if feasible:
                c_low = c_mid
            else:
                c_high = c_mid
        c = c_low
        feasible, values = check_feasibility(c)
        
        if not feasible:
             # Fallback to very small c
             c = 0.1
             feasible, values = check_feasibility(c)
             if not feasible:
                 domain = np.array([1], dtype=np.int64)
                 values = np.array([0.0], dtype=np.float64)
                 return domain, values

    return domain, values

# EVOLVE-BLOCK-END
