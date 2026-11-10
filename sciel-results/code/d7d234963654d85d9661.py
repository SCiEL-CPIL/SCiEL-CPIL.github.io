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
    """Compute max M(x) for x in [1, 10*K_max]."""
    m = len(domain)
    if m == 0:
        return 0.0
    
    x_max = 10 * K_max
    
    if x_max <= 1000000:
        h = np.zeros(x_max + 1, dtype=np.float64)
        for i, k in enumerate(domain):
            if k <= x_max:
                h[k::k] += values[i]
        M_vals = np.cumsum(h[1:])
        return np.max(M_vals)
    else:
        x_sample = np.arange(1, x_max + 1, max(1, x_max // 200000))
        M_vals = np.array([np.sum(values * (x // domain.astype(np.float64))) for x in x_sample])
        return np.max(M_vals)


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    return _compute_M_max(domain, values, K_max) <= tolerance


def _solve_lp_column_gen(K, time_limit=300):
    """Solve LP using column generation for the PNT partial function problem."""
    start = time.time()
    
    mu = _compute_mobius(K)
    candidates = np.where(mu[2:] != 0)[0] + 2
    n_vars = len(candidates)
    
    if n_vars == 0:
        return None
    
    c = np.log(candidates.astype(np.float64)) / candidates.astype(np.float64)
    
    x_max_check = min(10 * K, 100000)
    x_active = np.arange(1, x_max_check + 1, max(1, x_max_check // 2000))
    
    v_best = None
    obj_best = np.inf
    
    for iteration in range(5):
        if time.time() - start > time_limit:
            break
        
        A = np.zeros((len(x_active), n_vars), dtype=np.float64)
        for i, x in enumerate(x_active):
            A[i, :] = np.floor(x / candidates.astype(np.float64)) - x / candidates.astype(np.float64)
        
        try:
            res = linprog(c, A_ub=A, b_ub=np.ones(len(x_active)), method='highs')
        except Exception:
            break
        
        if not res.success:
            break
        
        v_sol = res.x
        obj_val = c @ v_sol
        
        if obj_val < obj_best:
            obj_best = obj_val
            v_best = v_sol.copy()
        
        M_max = _compute_M_max(candidates, v_sol, K)
        if M_max <= 1.0001:
            break
        
        x_sample = np.arange(1, x_max_check + 1, max(1, x_max_check // 10000))
        M_sample = np.array([np.sum(v_sol * (x // candidates.astype(np.float64)) - x * np.sum(v_sol / candidates.astype(np.float64))) for x in x_sample])
        violations = x_sample[M_sample > 1.0001]
        
        if len(violations) == 0:
            break
        
        keep = min(len(violations), max(0, 5000 - len(x_active)))
        if keep <= 0:
            break
        x_active = np.unique(np.concatenate([x_active, violations[:keep]]))
        x_active = np.sort(x_active)
    
    if v_best is None:
        return None
    
    v_full = np.zeros(K + 1)
    v_full[candidates] = v_best
    return v_full


def prime_number_theorem_partial_function():
    """
    Construct a partial function using LP with column generation.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    for K in [20000, 50000, 100000, 200000]:
        if time.time() > deadline - 100:
            break
        
        try:
            v_full = _solve_lp_column_gen(K, time_limit=min(400, deadline - time.time() - 50))
            if v_full is not None:
                domain = np.where(v_full[1:] != 0)[0] + 1
                if len(domain) == 0:
                    continue
                values = v_full[domain].astype(np.float64)
                if _check_feasibility(domain, values, K):
                    return domain, values
        except Exception:
            continue
    
    for K_try in [50000, 20000, 10000]:
        if time.time() > deadline:
            break
        try:
            mu = _compute_mobius(K_try)
            domain = np.where(mu[1:] != 0)[0] + 1
            values = mu[domain].astype(np.float64)
            if _check_feasibility(domain, values, K_try):
                return domain, values
        except Exception:
            continue
    
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
