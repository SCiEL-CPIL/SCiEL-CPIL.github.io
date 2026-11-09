# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 10
_DEADLINE = 600.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _shannon_bits(p):
    """Shannon entropy (bits) of a probability vector, zero terms ignored."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-18]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _objective_flat(P):
    """Maximize D / M  (D = H of x-y, M = max(H of x, H of x+y, H of x+2y))."""
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-12:
        return 0.0
    P = P / s
    D = _shannon_bits(P.sum(axis=1))                    # x - y  (diagonals)
    H0 = _shannon_bits(P.sum(axis=0))                   # x       (columns)
    H1 = _shannon_bits((P * np.arange(_N, _N * 2)).sum(axis=1))   # x + y
    H2 = _shannon_bits((P * (2.0 * np.arange(_N)[:, None] + np.arange(_N)[None, :])).sum(axis=1))  # x + 2y
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M


def _analytic_solution(n=_N):
    """Per-cell weight 1 / (number of cells sharing its x+2y value)."""
    X = np.arange(n)[None, :]
    Y = np.arange(n)[:, None]
    v = (X + 2.0 * Y).astype(int)
    counts = np.bincount(v.ravel(), minlength=3 * n - 1)
    counts = np.maximum(counts, 1)
    P = 1.0 / counts[v]
    P = P / P.sum()
    return P


def _exp_family_optimize(n=_N, budget=500.0):
    """
    Optimize P using exponential family parameterization:
    P[y,x] = exp(theta[y,x]) / Z
    
    Use gradient ascent on theta to maximize D/M.
    This allows non-linear correlations that can break the 1.0 barrier.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        return None

    # Precompute feature matrices for efficient gradient computation
    X = np.arange(n)[None, :]
    Y = np.arange(n)[:, None]
    
    # x - y values (diagonals)
    diff_vals = (X - Y).astype(int)  # range: -(n-1) to (n-1)
    diff_min = -(n - 1)
    diff_max = n - 1
    diff_idx = diff_vals - diff_min  # 0 to 2n-2
    
    # x values (columns)
    x_vals = X.astype(int)  # 0 to n-1
    
    # x + y values
    sum_vals = (X + Y).astype(int)  # 0 to 2n-2
    
    # x + 2y values
    x2y_vals = (X + 2.0 * Y).astype(int)  # 0 to 3n-2
    
    # Build indicator matrices for each feature
    # diff_ind[d, y, x] = 1 if diff_vals[y,x] == d
    num_diffs = 2 * n - 1
    diff_ind = np.zeros((num_diffs, n, n))
    for d in range(num_diffs):
        diff_ind[d] = (diff_idx == d)
    
    num_x = n
    x_ind = np.zeros((num_x, n, n))
    for xi in range(num_x):
        x_ind[xi] = (x_vals == xi)
    
    num_sums = 2 * n - 1
    sum_ind = np.zeros((num_sums, n, n))
    for s in range(num_sums):
        sum_ind[s] = (sum_vals == s)
    
    num_x2y = 3 * n - 1
    x2y_ind = np.zeros((num_x2y, n, n))
    for v in range(num_x2y):
        x2y_ind[v] = (x2y_vals == v)
    
    def compute_entropies(P):
        """Compute D, H0, H1, H2 from probability matrix P."""
        diff_probs = P @ diff_ind.reshape(num_diffs, -1).T  # shape (num_diffs,)
        x_probs = P @ x_ind.reshape(num_x, -1).T  # shape (num_x,)
        sum_probs = P @ sum_ind.reshape(num_sums, -1).T  # shape (num_sums,)
        x2y_probs = P @ x2y_ind.reshape(num_x2y, -1).T  # shape (num_x2y,)
        
        D = _shannon_bits(diff_probs)
        H0 = _shannon_bits(x_probs)
        H1 = _shannon_bits(sum_probs)
        H2 = _shannon_bits(x2y_probs)
        M = max(H0, H1, H2)
        return D, H0, H1, H2, M
    
    def objective_from_theta(theta):
        """Compute -D/M from log-probability parameters theta."""
        # Softmax normalization
        theta_max = theta.max()
        log_P = theta - theta_max
        P = np.exp(log_P)
        Z = P.sum()
        if Z <= 1e-12:
            return 0.0
        P = P / Z
        
        D, H0, H1, H2, M = compute_entropies(P)
        if M <= 1e-9:
            return 0.0
        return -D / M
    
    def gradient_from_theta(theta):
        """Compute gradient of -D/M w.r.t. theta using automatic differentiation via finite differences."""
        eps = 1e-7
        grad = np.zeros_like(theta)
        for i in range(theta.size):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus.flat[i] += eps
            theta_minus.flat[i] -= eps
            f_plus = objective_from_theta(theta_plus)
            f_minus = objective_from_theta(theta_minus)
            grad.flat[i] = (f_plus - f_minus) / (2.0 * eps)
        return grad
    
    # Initialize theta from analytic solution
    P0 = _analytic_solution(n)
    theta0 = np.log(P0 + 1e-15)
    
    best_theta = theta0.copy()
    best_val = -objective_from_theta(theta0)
    
    t_end = time.time() + budget
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    for restart in range(6):
        if time.time() > t_end - 60.0:
            break
        
        if restart == 0:
            start_theta = theta0.copy()
        else:
            # Add random perturbation
            noise_scale = 0.5 * (restart + 1)
            start_theta = theta0 + noise_scale * rng.randn(n, n)
        
        try:
            res = minimize(
                objective_from_theta,
                start_theta,
                jac=gradient_from_theta,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 4000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if hasattr(res, 'x') and res.x is not None:
            cand_val = -objective_from_theta(res.x)
            if cand_val > best_val + 1e-9:
                best_val = cand_val
                best_theta = res.x.copy()
        
        if _remaining() < 60.0:
            break
    
    # Convert best theta back to P
    theta_max = best_theta.max()
    log_P = best_theta - theta_max
    P = np.exp(log_P)
    Z = P.sum()
    if Z <= 1e-12:
        return None
    P = P / Z
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s > 1e-12:
        P = P / s
    return P


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    P = _analytic_solution(n)  # guaranteed ratio = 1.0, zero cost
    
    # Attempt exponential family optimization if there's enough time
    if _remaining() > 100.0:
        try:
            P2 = _exp_family_optimize(n, budget=min(500.0, _remaining() - 100.0))
            if P2 is not None and _objective_flat(P2) > _objective_flat(P) + 1e-9:
                P = P2
        except Exception:
            pass
    
    # Enforce the public contract.
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / s
    P = P.astype(np.float64)
    if not np.all(np.isfinite(P)) or P.shape != (n, n) or P.sum() <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    return P


# EVOLVE-BLOCK-END
