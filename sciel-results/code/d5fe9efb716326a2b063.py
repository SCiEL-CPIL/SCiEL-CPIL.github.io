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


def _compute_M_max(domain, values, K_max):
    """Compute max M(x) for x in [1, 10*K_max] efficiently using divisor sums."""
    x_max = 10 * K_max
    m = len(domain)
    if m == 0 or x_max <= 0:
        return 0.0
    
    h = np.zeros(x_max + 1, dtype=np.float64)
    for i in range(m):
        k = domain[i]
        v = values[i]
        if v != 0:
            h[k::k] += v
    
    M_vals = np.cumsum(h[1:])
    return float(np.max(M_vals))


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    m_max = _compute_M_max(domain, values, K_max)
    return m_max <= tolerance


def _compute_metric(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    if len(domain) == 0:
        return 0.0
    return float(-np.sum(values * np.log(domain) / domain))


def _apply_clipping_and_normalization(domain, values):
    """Apply clipping to [-10, 10] and normalize so sum f(k)/k = 0."""
    values = np.clip(values, -10.0, 10.0)
    
    S = np.sum(values / domain)
    
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values[idx_1[0]] -= S
    else:
        domain = np.concatenate([[1], domain])
        values = np.concatenate([[-S], values])
        sort_idx = np.argsort(domain)
        domain = domain[sort_idx]
        values = values[sort_idx]
    
    return domain, values


def _solve_lp(K, deadline):
    """
    Solve the LP to maximize the metric after normalization.
    
    After normalization: f(1) = -sum_{k>1} f(k)/k
    Metric: a = -sum_{k>1} f(k) * log(k) / k
    Constraint: M(x) = sum_{k>1} f(k) * (floor(x/k) - x/k) <= 1.0001 for x in [1, 10K]
    """
    start_time = time.time()
    
    n_vars = K - 1  # f(2), f(3), ..., f(K)
    
    if n_vars <= 0:
        return None
    
    ks = np.arange(2, K + 1)
    
    # Objective: minimize sum f(k) * log(k) / k
    c = np.log(ks) / ks
    
    bounds = [(-10.0, 10.0)] * n_vars
    
    x_max = 10 * K
    n_constraints = x_max
    
    # Build constraint matrix efficiently
    # A[i, j] = floor((i+1)/(j+2)) - (i+1)/(j+2) = -((i+1) % (j+2)) / (j+2)
    
    # Memory check: n_constraints * n_vars * 8 bytes
    # For K=5000: 50000 * 4999 * 8 = ~2GB - too large!
    # For K=2000: 20000 * 1999 * 8 = ~320MB - might work
    # For K=1000: 10000 * 999 * 8 = ~80MB - should work
    
    mem_needed = n_constraints * n_vars * 8
    if mem_needed > 500 * 1024 * 1024:  # 500MB limit
        # Use a different approach or smaller K
        return None
    
    A_ub = np.empty((n_constraints, n_vars), dtype=np.float64)
    
    # Build column by column
    for j in range(n_vars):
        k = j + 2
        # For x = 1, 2, ..., x_max: coefficient = -(x % k) / k
        xs = np.arange(1, x_max + 1)
        A_ub[:, j] = -(xs % k) / k
        
        if time.time() > deadline - 30:
            return None
    
    b_ub = np.full(n_constraints, 1.0001)
    
    # Solve
    time_left = deadline - time.time() - 10
    if time_left <= 5:
        return None
    
    try:
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'time_limit': time_left}
        )
    except Exception:
        return None
    
    if not result.success:
        return None
    
    f_vals = result.x
    
    # Construct solution
    domain = np.arange(1, K + 1)
    values = np.zeros(K + 1, dtype=np.float64)
    values[2:] = f_vals
    
    S = np.sum(f_vals / ks)
    values[1] = -S
    values = np.clip(values, -10.0, 10.0)
    
    # Re-normalize
    S2 = np.sum(values / domain)
    values[0] = -S2
    values = np.clip(values, -10.0, 10.0)
    
    # Remove zeros
    mask = np.abs(values[1:]) > 1e-10
    domain_out = np.concatenate([[1], ks[mask]])
    values_out = np.concatenate([[values[1]], f_vals[mask]])
    
    if not _check_feasibility(domain_out, values_out, K):
        return None
    
    metric = _compute_metric(domain_out, values_out)
    return domain_out, values_out, metric


def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580
    
    best_result = None
    best_metric = -np.inf
    
    # Try LP with manageable K values
    for K_try in [1000, 2000, 5000]:
        if time.time() > deadline - 60:
            break
        
        result = _solve_lp(K_try, deadline)
        if result is not None:
            domain, values, metric = result
            if metric > best_metric:
                best_metric = metric
                best_result = (domain, values)
        
        if time.time() > deadline - 30:
            break
    
    if best_result is None:
        # Fallback to Möbius
        K = 200000
        mu = _compute_mobius(K)
        domain = np.where(mu[1:] != 0)[0] + 1
        values = mu[domain].astype(np.float64)
        
        if not _check_feasibility(domain, values, K):
            for K_try in [100000, 50000, 20000, 10000, 5000, 2000, 1000]:
                if time.time() > deadline:
                    break
                mu_try = _compute_mobius(K_try)
                domain_try = np.where(mu_try[1:] != 0)[0] + 1
                values_try = mu_try[domain_try].astype(np.float64)
                if _check_feasibility(domain_try, values_try, K_try):
                    domain, values, K = domain_try, values_try, K_try
                    break
            else:
                domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
                values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        
        return domain, values
    
    return best_result[0], best_result[1]


# EVOLVE-BLOCK-END
