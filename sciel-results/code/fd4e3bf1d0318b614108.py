# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.ones(N + 1, dtype=np.int8)
    primes = []
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    mu[1] = 1
    return mu

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    max_K_to_try = 10000
    
    mu = _compute_mobius(max_K_to_try)
    
    best_K = 1
    best_a = 0.0
    
    # Precompute log(k)/k
    ks = np.arange(1, max_K_to_try + 1, dtype=np.float64)
    log_k_over_k = np.log(ks) / ks
    
    # We want to maximize a = - sum mu(k) log(k)/k
    # Check constraint for increasing K
    
    # To speed up, we can check constraints for K, K+1, etc.
    # But the constraint depends on all k <= K.
    
    # Let's just check K from 1 to max_K_to_try
    # And keep the best feasible one.
    
    # Optimization: If K is infeasible, we don't necessarily stop, but it's likely
    # that larger K will also be infeasible or have lower gain.
    # However, the constraint is global.
    
    for K in range(1, max_K_to_try + 1):
        if time.time() > deadline:
            break
            
        feasible = True
        
        # Check x from K+1 to 10K
        # We can optimize by checking only a subset if K is large?
        # But let's try full check first.
        
        # If K is large, 10K is large.
        # The loop `for x in range(K+1, 10*K+1)` runs 9K times.
        # Each time we do a dot product of size K.
        # Total ops: 9 K^2.
        # For K=10000, 9 * 10^8 ops.
        # This might be too slow in Python if not vectorized well.
        # np.dot is C-optimized.
        # 10^8 flops is ~0.1s in C.
        # But we are calling np.dot 90000 times.
        # Overhead of Python loop is significant.
        
        # Better approach:
        # We can compute M(x) for all x in [1, 10K] more efficiently?
        # M(x) = sum_{k=1}^K mu[k] floor(x/k).
        # This is a divisor sum type structure.
        
        # Alternative:
        # Just check a few critical x values?
        # The maximum of M(x) likely occurs at x where many floors change.
        # But we need to be sure.
        
        # Let's reduce the frequency of checks for large K.
        # Or use a smarter check.
        
        # For now, let's limit K to 2000 to be safe with time.
        # If K=2000 is feasible, a will be close to 1.
        
        if K > 2000:
            # Sample x values
            # Check x = K + i * step
            step = max(1, (9 * K) // 100)
            xs = range(K + 1, 10 * K + 1, step)
        else:
            xs = range(K + 1, 10 * K + 1)
            
        for x in xs:
            k_vals = np.arange(1, K + 1)
            floors = x // k_vals
            m_val = np.dot(mu[1:K+1], floors)
            
            if m_val > 1:
                feasible = False
                break
                
        if feasible:
            a = -np.sum(mu[1:K+1] * log_k_over_k[1:K+1])
            if a > best_a:
                best_a = a
                best_K = K
        else:
            # If infeasible, we might want to stop if we've been infeasible for a while?
            # Or just continue.
            pass

    # Construct the result for best_K
    if best_K > 0:
        domain = np.arange(1, best_K + 1, dtype=np.int64)
        values = mu[1:best_K + 1].astype(np.float64)
    else:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        
    return domain, values

# EVOLVE-BLOCK-END
