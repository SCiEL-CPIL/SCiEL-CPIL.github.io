# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _entropy_from_pooled(pooled):
    """Compute Shannon entropy in bits from a 1-D array of pooled weights."""
    total = pooled.sum()
    if total <= 1e-15:
        return 0.0
    probs = pooled[pooled > 1e-15] / total
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

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
    d_pooled = np.bincount(d_vals, weights=w, minlength=19)
    D = _entropy_from_pooled(d_pooled)
    
    # M = max of H of x, x+y, x+2y
    x_pooled = np.bincount(x_flat, weights=w, minlength=10)
    xy_pooled = np.bincount(x_flat + y_flat, weights=w, minlength=19)
    x2y_pooled = np.bincount(x_flat + 2 * y_flat, weights=w, minlength=28)
    
    h0 = _entropy_from_pooled(x_pooled)
    h1 = _entropy_from_pooled(xy_pooled)
    h2 = _entropy_from_pooled(x2y_pooled)
    M = max(h0, h1, h2)
    
    if M <= 1e-15:
        return 0.0
    return D / M

def _neg_ratio(theta, best_score):
    """Objective function: negative of D/M ratio."""
    # Convert theta to probabilities via softmax
    theta = theta - theta.max()  # numerical stability
    exp_theta = np.exp(theta)
    P = exp_theta / exp_theta.sum()
    P = P.reshape(10, 10)
    score = _compute_ratio(P)
    if score > best_score[0]:
        best_score[0] = score
        best_score[1][:] = P
    return -score

def _optimize_P():
    """Optimize P to maximize D/M using multiple strategies."""
    deadline = time.time() + 590  # safety margin before 630s limit
    
    n = 10
    best_P = np.ones((n, n)) / (n * n)
    best_score = [0.0, best_P.copy()]
    
    # Strategy 1: Start from uniform, Nelder-Mead
    theta0 = np.log(best_P.flatten())
    
    try:
        result = minimize(
            lambda t: _neg_ratio(t, best_score),
            theta0,
            method='Nelder-Mead',
            options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_score[1]
    
    # Strategy 2: Exploit arithmetic structure - concentrate on lines of constant x+2y
    # to minimize H(x+2y), which allows H(x-y) to dominate the ratio.
    # For each value c of x+2y, the cells (x,y) satisfying x+2y=c form a line.
    # We want to spread weight along these lines to maximize H(x-y).
    
    # Precompute the structure
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    x2y_vals = x_flat + 2 * y_flat
    xy_vals = x_flat + y_flat
    x_minus_y_vals = x_flat - y_flat
    
    # Group indices by x+2y value
    from collections import defaultdict
    x2y_groups = defaultdict(list)
    for i in range(100):
        x2y_groups[x2y_vals[i]].append(i)
    
    # Try concentrated distributions on specific x+2y lines
    for c in range(28):
        if time.time() > deadline:
            break
        indices = x2y_groups.get(c, [])
        if len(indices) < 2:
            continue
        
        # Start with uniform weight on this line
        P_line = np.zeros(100)
        P_line[indices] = 1.0 / len(indices)
        theta_line = np.log(P_line + 1e-15)
        
        try:
            result = minimize(
                lambda t: _neg_ratio(t, best_score),
                theta_line,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_score[1]
    
    # Strategy 3: Random restarts with diverse initializations
    np.random.seed(123)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Create a random sparse-ish initialization
        theta_rand = np.random.randn(100) * 1.0
        # Occasionally bias towards structured patterns
        if restart % 3 == 0:
            # Diagonal bias
            theta_rand += np.diag(np.ones(10)).flatten() * 0.5
        elif restart % 3 == 1:
            # Row bias
            theta_rand += np.tile(np.random.randn(10), 10) * 0.3
        else:
            # Column bias
            theta_rand += np.repeat(np.random.randn(10), 10) * 0.3
        
        try:
            result = minimize(
                lambda t: _neg_ratio(t, best_score),
                theta_rand,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
        except Exception:
            pass
    
    return best_score[1]

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
