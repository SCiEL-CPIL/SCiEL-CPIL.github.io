# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for i.i.d. draws from discrete measure."""
    n = len(support)
    w_sum = weights.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Pairwise sums
    pair_sums = support[:, None] + support[None, :]  # (n, n)
    ww = w[:, None] * w[None, :]  # (n, n)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            mask = pair_sums < threshold
            inner = np.sum(ww * mask)
            P += w[k] * w[l] * inner
    return P

def _neg_P(params, n):
    """Negative P for minimization."""
    support = np.abs(params[:n])
    support = np.sort(support)
    weights = np.abs(params[n:])
    w_sum = weights.sum()
    if w_sum < 1e-15:
        return 1.0
    P = _compute_P(support, weights)
    return -P

def _optimize_n_points(n, time_budget=120):
    """Optimize for n support points. Returns (support, weights, P) or None."""
    start = time.time()
    best_P = 0.0
    best_params = None
    
    np.random.seed(123)
    n_restarts = 30
    
    for r in range(n_restarts):
        if time.time() - start > time_budget:
            break
        
        s_init = np.sort(np.random.uniform(0, 4, n))
        w_init = np.random.dirichlet(np.ones(n))
        params_init = np.concatenate([s_init, w_init])
        
        bounds = [(0, 8)] * n + [(1e-6, 1)] * n
        
        try:
            result = differential_evolution(
                _neg_P, bounds, args=(n,),
                maxiter=50, popsize=12, tol=1e-5,
                seed=1000 + r, init='sobol',
                polish=False
            )
            P_val = -result.fun
            if P_val > best_P:
                best_P = P_val
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return None
    
    support = np.sort(np.abs(best_params[:n]))
    weights = np.abs(best_params[n:])
    return support, weights, best_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    deadline = 580  # seconds, leave margin from 630
    start_time = time.time()
    
    best_support = None
    best_weights = None
    best_P = 0.0
    
    # Try 3 points first (fastest), then 4, then 5 if time permits
    for n in [3, 4, 5]:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 40:
            break
        
        time_budget = min(150, remaining - 30)
        if time_budget < 20:
            break
        
        result = _optimize_n_points(n, time_budget=time_budget)
        if result is not None:
            support, weights, P = result
            if P > best_P:
                best_P = P
                best_support = support
                best_weights = weights
    
    # Fallback if optimization didn't find anything good
    if best_support is None or best_P < 0.3:
        # Known decent 3-point distribution
        support = np.array([0.0, 1.0, 4.0], dtype=np.float64)
        weights = np.array([0.4, 0.4, 0.2], dtype=np.float64)
        P = _compute_P(support, weights)
        if P > best_P:
            best_P = P
            best_support = support
            best_weights = weights
    
    if best_support is None:
        support = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float64) / 3
    else:
        support = best_support.astype(np.float64)
        weights = best_weights.astype(np.float64)
    
    # Final validation
    support = np.maximum(support, 0.0)
    weights = np.maximum(weights, 0.0)
    
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        support = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float64) / 3
    
    if weights.sum() <= 1e-9:
        weights = np.ones(len(support), dtype=np.float64) / len(support)
    
    return support, weights
# EVOLVE-BLOCK-END
