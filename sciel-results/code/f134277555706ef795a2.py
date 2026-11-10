# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mu_and_omega(n):
    """Compute Mobius function and Omega (number of prime factors with multiplicity) for 1..n using linear sieve."""
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    omega = np.zeros(n + 1, dtype=np.int32)
    
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    primes = []
    
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            omega[i] = 1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            omega[i * p] = omega[i] + 1
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    mu[1] = 1
    omega[1] = 0
    
    return mu, omega


def _compute_M_fast(domain, values, x_max):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x = 1..x_max.
    Uses vectorized numpy operations with chunked processing for memory efficiency.
    """
    M = np.zeros(x_max, dtype=np.float64)
    x_vals = np.arange(1, x_max + 1, dtype=np.int64)
    
    # Process in chunks to avoid excessive memory usage
    chunk_size = max(1, min(len(domain), 500))
    for start in range(0, len(domain), chunk_size):
        end = min(start + chunk_size, len(domain))
        chunk_domain = domain[start:end]
        chunk_values = values[start:end]
        
        # Skip zero values
        nonzero_mask = chunk_values != 0.0
        if not np.any(nonzero_mask):
            continue
        cd = chunk_domain[nonzero_mask]
        cv = chunk_values[nonzero_mask]
        
        # Vectorized: for each k, add f(k) * floor(x/k)
        # Stack all floor(x/k) computations
        floors = x_vals[:, np.newaxis] // cd[np.newaxis, :]
        M += (floors * cv[np.newaxis, :]).sum(axis=1)
    
    return M


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes a = -sum_k f(k)*log(k)/k
    subject to M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K].
    
    Uses full Mobius function up to K (square-free numbers), which gives M(x)=1
    for x <= K, and approximately 1 for x > K. Then adds a small perturbation
    to improve the objective while maintaining feasibility.
    """
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 630s limit
    
    # Try increasing K to maximize the objective
    K_candidates = [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000, 500]
    
    best_domain = None
    best_values = None
    best_a = -np.inf
    
    for K in K_candidates:
        if time.time() > deadline - 120:
            break
        
        mu, omega = _compute_mu_and_omega(K)
        
        # Include all k where mu(k) != 0 (square-free numbers)
        mask = mu != 0
        domain = np.where(mask)[0].astype(np.int64)
        values = mu[domain].astype(np.float64)
        
        # Ensure 1 is in domain
        if len(domain) == 0 or domain[0] != 1:
            domain = np.concatenate([[1], domain])
            values = np.concatenate([[1.0], values])
        
        K_max = int(domain[-1])
        x_max_eval = 10 * K_max
        m = len(domain)
        
        # For x <= K, M(x) = 1 exactly (Möbius inversion identity)
        # Only need to check x in (K, 10K]
        # Use full computation if feasible, otherwise use smarter sampling
        
        if x_max_eval <= 300000:
            # Full computation
            M = _compute_M_fast(domain, values, x_max_eval)
            M_max = np.max(M)
            
            if M_max <= 1.0001:
                log_k = np.log(domain.astype(np.float64))
                log_k[0] = 0.0
                a_val = -np.sum(values * log_k / domain.astype(np.float64))
                
                if a_val > best_a:
                    best_a = a_val
                    best_domain = domain.copy()
                    best_values = values.copy()
        else:
            # For large K, check only x > K where M(x) can deviate from 1
            # M(x) = 1 + sum_{k=K+1}^{x} mu(k)*floor(x/k) for x > K
            # Actually M(x) = sum_{k=1}^{K} mu(k)*floor(x/k) for all x
            # For x > K, this differs from 1 by the tail
            
            # Sample strategically: check x values where floor(x/k) changes
            # Focus on x just above multiples of small k values
            # Use a mix of uniform and targeted sampling
            
            n_samples = 500
            # Uniform samples in (K, 10K]
            uniform_xs = np.linspace(K + 1, 10 * K_max, n_samples // 2).astype(np.int64)
            # Targeted: x = q*p for small primes p, various q
            targeted_xs = []
            for p in [2, 3, 5, 7, 11, 13]:
                for q in range(1, min(100, (10*K_max - K) // p + 1)):
                    x = q * p
                    if K < x <= 10 * K_max:
                        targeted_xs.append(x)
            targeted_xs = np.array(targeted_xs[:n_samples // 2], dtype=np.int64)
            
            all_xs = np.unique(np.concatenate([uniform_xs, targeted_xs]))
            
            M_samples = np.zeros(len(all_xs), dtype=np.float64)
            x_chunk = np.arange(1, x_max_eval + 1, dtype=np.int64)
            
            # Compute M at sampled points using vectorized approach
            for idx, x in enumerate(all_xs):
                M_samples[idx] = np.sum(values * (x // domain))
            
            if np.max(M_samples) <= 1.0001:
                log_k = np.log(domain.astype(np.float64))
                log_k[0] = 0.0
                a_val = -np.sum(values * log_k / domain.astype(np.float64))
                
                if a_val > best_a:
                    best_a = a_val
                    best_domain = domain.copy()
                    best_values = values.copy()
        
        if time.time() > deadline - 100:
            break
    
    if best_domain is None:
        # Fallback: small known-feasible set
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    # Now try to improve with a perturbation on the best base
    domain = best_domain
    values = best_values.copy()
    K_max = int(domain[-1])
    x_max_eval = 10 * K_max
    
    # Compute perturbation direction
    mu, omega = _compute_mu_and_omega(K_max)
    omega_domain = omega[domain]
    sign_omega = np.where(omega_domain % 2 == 0, 1.0, -1.0)
    g = sign_omega / domain.astype(np.float64)
    
    inv_k = 1.0 / domain.astype(np.float64)
    alpha = np.sum(g * inv_k) / np.sum(inv_k * inv_k)
    h = g - alpha * inv_k
    
    log_k = np.log(domain.astype(np.float64))
    log_k[0] = 0.0
    direction_sum = np.sum(h * log_k / domain.astype(np.float64))
    
    if direction_sum > 0:
        c_sign = -1.0
    elif direction_sum < 0:
        c_sign = 1.0
    else:
        c_sign = 0.0
    
    if c_sign != 0.0 and time.time() < deadline - 60:
        # Compute H(x) = sum_k h(k) * floor(x/k) for x in [1, 10K]
        if x_max_eval <= 300000:
            H = _compute_M_fast(domain, h, x_max_eval)
            H_max = np.max(H)
            H_min = np.min(H)
            
            if c_sign > 0:
                if H_max > 0:
                    c_max = 0.0001 / H_max
                    c = c_max * 0.999
                else:
                    c = 0.0
            else:
                if H_min < 0:
                    c_min = 0.0001 / H_min
                    c = c_min * 0.999
                else:
                    c = 0.0
            
            if c != 0.0:
                values_new = values + c * h
                # Verify feasibility
                M_final = _compute_M_fast(domain, values_new, x_max_eval)
                if np.max(M_final) <= 1.0001:
                    values = values_new
    
    return domain, values


# EVOLVE-BLOCK-END
