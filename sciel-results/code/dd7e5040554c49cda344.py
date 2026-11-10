# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def _sieve_mobius(n):
    """Compute Mobius function for 1..n."""
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    mu[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def prime_number_theorem_partial_function():
    """
    Maximize a = -sum_k f(k)*log(k)/k subject to M(x) <= 1.0001 for all x in [1, 10K].
    Uses LP with Möbius-like structure and large negative values for composite keys.
    """
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 630s limit

    K = 50
    K_max = K
    x_max = 10 * K_max  # 500
    n_vars = K  # f(1), f(2), ..., f(K)
    
    # Build constraint matrix for M(x) = sum_k f(k)*floor(x/k) <= 1.0001
    # For x in [1, 500]: A[x-1, k-1] = floor(x/k) for k in [1, K]
    A_rows = []
    for x in range(1, x_max + 1):
        row = np.zeros(n_vars)
        for k in range(1, K + 1):
            if k <= x:
                row[k - 1] = x // k
        A_rows.append(row)
    A_mat = np.array(A_rows, dtype=np.float64)
    b_mat = np.full(x_max, 1.0001, dtype=np.float64)
    
    # Also enforce sum_k f(k)/k = 0 (after f(1) adjustment, this is the effective constraint)
    # Actually the evaluator does: f(1) = f(1) - S where S = sum_k f(k)/k
    # So sum_k f(k)/k = 0 is enforced by the evaluator, not by us.
    # We just need M(x) <= 1.0001 for all x in [1, 10K].
    
    # Objective: maximize a = -sum_k f(k)*log(k)/k
    # c[k] = -log(k)/k for k in [1, K], c[0] = 0 (since log(1)=0)
    k_arr = np.arange(1, K + 1, dtype=np.float64)
    c_obj = -np.log(k_arr) / k_arr
    c_obj[0] = 0.0  # log(1) = 0
    
    # Bounds: f(k) in [-10, 10] (clipping range)
    bounds = [(-10.0, 10.0)] * n_vars
    
    # Solve LP: minimize -a = sum_k log(k)*f(k)/k subject to A*f <= 1.0001, -10 <= f <= 10
    result = linprog(c_obj, A_ub=A_mat, b_ub=b_mat, bounds=bounds, method='highs')
    
    if time.time() > deadline - 10:
        # Return Möbius as fallback
        mu = _sieve_mobius(K)
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[domain].astype(np.float64)
        return domain, values
    
    if not result.success:
        # Fallback to Möbius
        mu = _sieve_mobius(K)
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[domain].astype(np.float64)
        return domain, values
    
    f = result.x
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = f.astype(np.float64)
    
    # Verify feasibility
    x_vals = np.arange(1, x_max + 1, dtype=np.int64)
    M = np.zeros(x_max, dtype=np.float64)
    for k, fk in zip(domain, values):
        if abs(fk) > 1e-15:
            M += fk * (x_vals // k)
    if np.max(M) > 1.0001 + 1e-9:
        # Revert to Möbius
        mu = _sieve_mobius(K)
        values = mu[domain].astype(np.float64)
    
    return domain, values


# EVOLVE-BLOCK-END
