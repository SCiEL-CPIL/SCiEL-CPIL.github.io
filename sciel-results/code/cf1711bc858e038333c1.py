# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _entropy_from_values(values, weights):
    """Compute Shannon entropy in bits given values and their weights."""
    # Pool weights by value
    from collections import defaultdict
    d = defaultdict(float)
    for v, w in zip(values, weights):
        d[v] += w
    total = sum(d.values())
    if total <= 1e-15:
        return 0.0
    probs = [w / total for w in d.values() if w > 1e-15]
    if not probs:
        return 0.0
    return -sum(p * np.log2(p) for p in probs)

def _compute_ratio(P):
    """Compute D/M ratio for a 10x10 distribution P[y,x]."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    # D = H of (x - y)
    d_vals = x_flat - y_flat
    D = _entropy_from_values(d_vals, w)
    
    # M = max of H of x, x+y, x+2y
    h0 = _entropy_from_values(x_flat, w)
    h1 = _entropy_from_values(x_flat + y_flat, w)
    h2 = _entropy_from_values(x_flat + 2 * y_flat, w)
    M = max(h0, h1, h2)
    
    if M <= 1e-15:
        return 0.0
    return D / M

def _optimize_P():
    """Optimize P to maximize D/M using scipy's minimize (minimizing negative ratio)."""
    deadline = time.time() + 580  # safety margin before 630s limit
    
    # Start from uniform
    n = 10
    P0 = np.ones((n, n)) / (n * n)
    best_P = P0.copy()
    best_score = _compute_ratio(P0)
    
    # Parameterize: use log-probabilities for unconstrained optimization
    # p_i = exp(theta_i) / sum(exp(theta_j))
    # theta has 100 parameters, but we can reduce by 1 (gauge freedom)
    
    def neg_ratio(theta):
        # Convert theta to probabilities
        theta = theta - theta.max()  # numerical stability
        exp_theta = np.exp(theta)
        P = exp_theta / exp_theta.sum()
        P = P.reshape(10, 10)
        score = _compute_ratio(P)
        if score > best_score[0]:
            best_score[0] = score
            best_P[:] = P
        return -score
    
    best_score = [best_score]
    
    # Use Nelder-Mead
    theta0 = np.log(P0.flatten())
    
    try:
        result = minimize(
            neg_ratio,
            theta0,
            method='Nelder-Mead',
            options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        if _compute_ratio(result.x.reshape(10, 10)) > best_score[0]:
            best_P = result.x.reshape(10, 10)
            best_P = np.clip(best_P, 0, None)
            best_P = best_P / best_P.sum()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_P
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        # Random initial point
        theta_rand = np.random.randn(100) * 0.5 + np.log(P0.flatten())
        try:
            result = minimize(
                neg_ratio,
                theta_rand,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if _compute_ratio(result.x.reshape(10, 10)) > best_score[0]:
                best_P = result.x.reshape(10, 10)
                best_P = np.clip(best_P, 0, None)
                best_P = best_P / best_P.sum()
        except Exception:
            pass
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    start_time = time.time()
    
    P = _optimize_P()
    
    # Final safety checks
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        # Fallback to uniform
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / s
    
    # Ensure finite
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    
    return P

# EVOLVE-BLOCK-END
