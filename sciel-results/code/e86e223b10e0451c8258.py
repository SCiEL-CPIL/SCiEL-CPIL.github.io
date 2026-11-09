# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    s = support
    w = weights / weights.sum()
    
    # Pairwise sums: s_i + s_j for all pairs, weighted by w_i * w_j
    # We need: sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    
    # Precompute all pairwise sums and their weights
    # pair_sums[i,j] = s[i] + s[j], pair_weights[i,j] = w[i]*w[j]
    pair_sums = s[:, None] + s[None, :]  # shape (n, n)
    pair_weights = w[:, None] * w[None, :]  # shape (n, n)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Sum of pair_weights where pair_sums < threshold
            mask = pair_sums < threshold
            inner = pair_weights[mask].sum()
            P += w[k] * w[l] * inner
    return P


def _neg_P(params):
    """Negative objective for minimization. params = [support..., weights...]"""
    n_support = 3  # fixed for this design
    support = params[:n_support]
    weights = params[n_support:]
    
    # Enforce constraints: support >= 0, weights >= 0
    support = np.maximum(support, 0.0)
    weights = np.maximum(weights, 0.0)
    
    wsum = weights.sum()
    if wsum < 1e-9:
        return 1e10  # invalid
    
    return -_compute_P(support, weights)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    # Start with a good initial guess: {0, 1, K} with appropriate weights
    # The large K strategy: put most weight on 0 and 1, small weight on K
    # Optimize jointly
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Try different initial values of K and weight distributions
    K_values = [5, 10, 20, 50, 100, 200, 500, 1000]
    p2_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]  # weight on K
    
    for K in K_values:
        for p2 in p2_values:
            if time.time() > deadline - 60:  # safety margin
                break
            
            # Initial: support = [0, 1, K], weights = [p1, p1, p2] where p1 = (1-p2)/2
            p1 = (1.0 - p2) / 2.0
            init_support = np.array([0.0, 1.0, K], dtype=np.float64)
            init_weights = np.array([p1, p1, p2], dtype=np.float64)
            
            # Evaluate initial
            P_init = _compute_P(init_support, init_weights)
            if P_init > best_score:
                best_score = P_init
                best_support = init_support.copy()
                best_weights = init_weights.copy()
            
            # Try to optimize with SLSQP
            try:
                x0 = np.concatenate([init_support, init_weights])
                
                # Constraints: support >= 0, weights >= 0, sum(weights) = 1
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x[3:]) - 1.0, 
                     'jac': lambda x: np.array([0,0,0,1,1,1])},
                ]
                bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
                
                result = minimize(
                    _neg_P, x0, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                
                if result.success or result.fun < _neg_P(x0):
                    opt_support = np.maximum(result.x[:3], 0.0)
                    opt_weights = np.maximum(result.x[3:], 0.0)
                    wsum = opt_weights.sum()
                    if wsum > 1e-9:
                        opt_weights = opt_weights / wsum
                        P_opt = _compute_P(opt_support, opt_weights)
                        if P_opt > best_score:
                            best_score = P_opt
                            best_support = opt_support.copy()
                            best_weights = opt_weights.copy()
            except Exception:
                pass
    
    # Also try 4-point support: {0, a, 1, K}
    if time.time() < deadline - 120:
        for K in [10, 50, 200, 1000]:
            for a in [0.2, 0.5, 0.8]:
                if time.time() > deadline - 60:
                    break
                
                p2 = 0.15  # weight on K
                p3 = 0.1   # weight on a
                p0 = (1.0 - p2 - p3) / 2.0  # split between 0 and 1
                
                init_support = np.array([0.0, a, 1.0, K], dtype=np.float64)
                init_weights = np.array([p0, p3, p0, p2], dtype=np.float64)
                
                P_init = _compute_P(init_support, init_weights)
                if P_init > best_score:
                    best_score = P_init
                    best_support = init_support.copy()
                    best_weights = init_weights.copy()
    
    # Final refinement: try to optimize the best solution found
    if best_support is not None and time.time() < deadline - 30:
        n = len(best_support)
        try:
            x0 = np.concatenate([best_support, best_weights])
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x[n:]) - 1.0,
                 'jac': lambda x: np.concatenate([np.zeros(n), np.ones(n)])},
            ]
            bounds = [(0, None)] * (2 * n)
            
            result = minimize(
                _neg_P, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            if result.fun < _neg_P(x0):
                opt_support = np.maximum(result.x[:n], 0.0)
                opt_weights = np.maximum(result.x[n:], 0.0)
                wsum = opt_weights.sum()
                if wsum > 1e-9:
                    opt_weights = opt_weights / wsum
                    P_opt = _compute_P(opt_support, opt_weights)
                    if P_opt > best_score:
                        best_score = P_opt
                        best_support = opt_support.copy()
                        best_weights = opt_weights.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_support is None or best_weights is None:
        # Fallback to simple valid solution
        best_support = np.array([0.0, 1.0, 10.0], dtype=np.float64)
        best_weights = np.array([0.4, 0.4, 0.2], dtype=np.float64)
    
    # Final cleanup: ensure no negative values, normalize weights
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    wsum = best_weights.sum()
    if wsum < 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / wsum
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
