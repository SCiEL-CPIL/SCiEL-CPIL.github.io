# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    m = len(domain)
    if m == 0:
        return True
    
    x_max = 10 * K_max
    if x_max > 2000000:
        x_range = np.arange(1, x_max + 1, max(1, x_max // 200000))
    else:
        x_range = np.arange(1, x_max + 1)
    
    if x_max > 1000000:
        x_sample = x_range[::max(1, len(x_range)//50000)]
        M_vals = np.zeros(len(x_sample))
        for idx, x in enumerate(x_sample):
            M_vals[idx] = np.sum(values * (x // domain))
        return np.all(M_vals <= tolerance)
    else:
        h = np.zeros(x_max + 1, dtype=np.float64)
        for i, k in enumerate(domain):
            h[k::k] += values[i]
        M_vals = np.cumsum(h[1:])
        return np.all(M_vals <= tolerance)


def _solve_lp_for_K(K, deadline):
    """Solve LP for given K."""
    start_time = time.time()
    
    n_vars = K - 1
    k_vals = np.arange(2, K + 1, dtype=np.int64)
    k_float = k_vals.astype(np.float64)
    
    # Objective: minimize sum f(k) * log(k) / k
    c = np.log(k_float) / k_float
    
    # Constraints: sum f(k) * (floor(x/k) - x/k) <= 1 for all x in [1, 10*K]
    x_max = 10 * K
    
    # Use all x values if feasible, otherwise sample
    if x_max <= 200000:
        x_vals = np.arange(1, x_max + 1)
    else:
        n_samples = min(200000, x_max)
        x_vals = np.linspace(1, x_max, n_samples, dtype=np.int64)
    
    n_constraints = len(x_vals)
    
    # Build constraint matrix in chunks to manage memory
    A_ub_list = []
    chunk_size = 5000
    for start in range(0, n_constraints, chunk_size):
        end = min(start + chunk_size, n_constraints)
        x_chunk = x_vals[start:end]
        
        # For each x in chunk, compute floor(x/k) - x/k for all k
        # Shape: (chunk_len, n_vars)
        A_chunk = (x_chunk[:, None].astype(np.int64) // k_vals[None, :]).astype(np.float64) - x_chunk[:, None].astype(np.float64) / k_float[None, :]
        A_ub_list.append(A_chunk)
        
        if time.time() > deadline - 30:
            return None
    
    if not A_ub_list:
        return None
    
    A_ub = np.vstack(A_ub_list)
    b_ub = np.ones(n_constraints)
    
    # Bounds
    bounds = [(-10, 10)] * n_vars
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            f_vals = result.x
            f1 = -np.sum(f_vals / k_float)
            domain = np.concatenate([[1], k_vals])
            values = np.concatenate([[f1], f_vals])
            
            if _check_feasibility(domain, values, K):
                return domain, values
            else:
                return None
        else:
            return None
    except Exception:
        return None


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes the PNT-related objective.
    Uses LP to find optimal values.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    # Try different K values
    for K in [20000, 15000, 10000, 8000, 5000, 3000, 2000, 1000]:
        if time.time() > deadline - 120:
            break
        
        result = _solve_lp_for_K(K, deadline)
        if result is not None:
            return result
    
    # Fallback to Möbius
    K = 200000
    mu = _compute_mobius(K)
    domain = np.where(mu[1:] != 0)[0] + 1
    values = mu[domain].astype(np.float64)
    
    if _check_feasibility(domain, values, K):
        return domain, values
    
    # Final fallback
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
