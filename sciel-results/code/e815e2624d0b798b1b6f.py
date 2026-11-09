import numpy as np
import time
from scipy.optimize import minimize

def get_points_from_weights(weights):
    """
    weights: array of 20 positive numbers.
    Returns (20, 2) array of points.
    """
    w_sum = np.sum(weights)
    if w_sum <= 0:
        w_sum = 1e-6
    angles = np.cumsum(weights / w_sum * 2 * np.pi)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def params_to_weights(params):
    """
    Convert 20 parameters to 20 positive weights.
    Uses softmax to ensure positivity and sum to 1.
    """
    # Subtract max for numerical stability
    shifted = params - np.max(params)
    exp_params = np.exp(shifted)
    w = exp_params / np.sum(exp_params)
    return w * 2 * np.pi

def get_points(params):
    w = params_to_weights(params)
    angles = np.cumsum(w)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def vertex_score(v_idx, points):
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    if sorted_dists[0] < 0.05:
        return 0.0
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def soft_min(scores, temperature=0.05):
    """
    Approximate min using log-sum-exp with negative scores.
    min(s) ≈ -T * log(sum(exp(-s/T)))
    """
    if len(scores) == 0:
        return 0.0
    neg_scores = -scores / temperature
    max_neg = np.max(neg_scores)
    sum_exp = np.sum(np.exp(neg_scores - max_neg))
    return -temperature * (np.log(sum_exp) + max_neg)

def min_vertex_score(params):
    points = get_points(params)
    scores = np.array([vertex_score(i, points) for i in range(20)])
    # Use soft min for smoothness
    return -soft_min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    n = 20
    # Initial guess: regular polygon
    # For softmax, all zeros gives uniform weights
    x0 = np.zeros(20)
    
    best_params = x0
    # For tracking best, we use the negative soft-min score
    best_score = min_vertex_score(x0)
    
    # Phase 1: Local Refinement with L-BFGS-B
    # Since we removed DE, we can afford more restarts or longer optimization
    rng = np.random.default_rng(42)
    
    restarts = [best_params]
    
    # Generate random restarts
    # Softmax parameters are scale-invariant, so range doesn't matter as much,
    # but we keep them in a reasonable range to avoid overflow/underflow in exp
    for i in range(20):
        if time.time() > deadline - 20:
            break
        x_rand = rng.uniform(-5.0, 5.0, size=20)
        restarts.append(x_rand)
        
    for x_init in restarts:
        if time.time() > deadline - 10:
            break
        try:
            # Use Nelder-Mead or L-BFGS-B?
            # Soft min is smooth, so L-BFGS-B should work well.
            res = minimize(
                min_vertex_score,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    return get_points(best_params)
