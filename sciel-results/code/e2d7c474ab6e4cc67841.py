# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize, linprog


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy:
    1. Try a small number of support points (n=4) and optimize both support
       positions and weights using scipy.optimize.minimize (SLSQP + Nelder-Mead polish).
    2. Additionally, try a linear programming approach on a fixed sparse support grid
       to find globally optimal weight distributions, which may find better solutions
       than local optimization alone.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    start_time = time.time()
    deadline = start_time + 570  # 570s safety margin before 630s kill
    
    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 0.0
        w = weights / w_sum
        
        # Compute all pairwise sums s_i + s_j
        s_i = support[:, np.newaxis]
        s_j = support[np.newaxis, :]
        pair_sums = s_i + s_j  # shape (n, n)
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                mask = pair_sums < threshold
                inner = np.sum(w[np.newaxis, :] * w[:, np.newaxis] * mask)
                P += w[k] * w[l] * inner
        
        return P
    
    def neg_P(params):
        """Negative of P for minimization. params = [s0, s1, s2, s3, w0, w1, w2, w3]"""
        support = np.maximum(params[:4], 0.0)
        weights = np.maximum(params[4:], 0.0)
        
        if np.sum(weights) < 1e-12:
            return 1e10
        
        return -compute_P(support, weights)
    
    # Initial guesses: various starting points
    inits = [
        # Uniform on {0, 1, 2, 3}
        np.array([0.0, 1.0, 2.0, 3.0, 0.25, 0.25, 0.25, 0.25]),
        # Concentrated at 0
        np.array([0.0, 5.0, 10.0, 20.0, 0.6, 0.2, 0.1, 0.1]),
        # Geometric-like
        np.array([0.0, 1.0, 3.0, 9.0, 0.5, 0.3, 0.15, 0.05]),
        # Spread out
        np.array([0.0, 2.0, 6.0, 18.0, 0.4, 0.3, 0.2, 0.1]),
        # Two-point-like
        np.array([0.0, 1.0, 100.0, 200.0, 0.8, 0.1, 0.05, 0.05]),
        # Another spread
        np.array([0.0, 0.5, 2.0, 8.0, 0.3, 0.3, 0.25, 0.15]),
        # More concentrated near 0
        np.array([0.0, 0.1, 0.5, 5.0, 0.7, 0.1, 0.1, 0.1]),
        # Logarithmic spacing
        np.array([0.0, 0.5, 2.0, 10.0, 0.5, 0.25, 0.15, 0.1]),
    ]
    
    best_params = None
    best_P = -1.0
    
    bounds = [(0, None)] * 4 + [(0, 1)] * 4
    
    # Phase 1: SLSQP optimization from multiple starting points
    for init in inits:
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(
                neg_P,
                init,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.fun < best_P or best_params is None:
                params = res.x
                support = np.maximum(params[:4], 0.0)
                weights = np.maximum(params[4:], 0.0)
                
                if np.sum(weights) > 1e-12:
                    P_val = -res.fun
                    if P_val > best_P:
                        best_P = P_val
                        best_params = params.copy()
        except Exception as e:
            pass
    
    # Phase 2: Nelder-Mead polish on the best solution found
    if best_params is not None and time.time() < deadline - 30:
        try:
            res_polish = minimize(
                neg_P,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_polish.fun < best_P:
                params = res_polish.x
                support = np.maximum(params[:4], 0.0)
                weights = np.maximum(params[4:], 0.0)
                
                if np.sum(weights) > 1e-12:
                    P_val = -res_polish.fun
                    if P_val > best_P:
                        best_P = P_val
                        best_params = params.copy()
        except Exception as e:
            pass
    
    # Phase 3: LP-based optimization on fixed sparse support grids
    # The key insight: for a FIXED support, P is a quartic form in w, not linear.
    # However, we can reformulate: fix support, and note that for each (k,l),
    # S(k,l) = sum_{i,j} w_i w_j * 1[s_i + s_j < 2*s_l - s_k] is quadratic in w.
    # P = sum_{k,l} w_k w_l S(k,l) is quartic. Not directly LP-able.
    #
    # Alternative: try a different approach. For a fixed support, we can use
    # coordinate descent or other methods. But let's try a simpler idea:
    # For a fixed support, maximize P by treating it as a non-linear program
    # with linear constraints (simplex). Use scipy.optimize.minimize with
    # constraints.
    
    def compute_P_fixed_support(weights, support):
        """Compute P for fixed support and variable weights."""
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 0.0
        w = weights / w_sum
        
        s_i = support[:, np.newaxis]
        s_j = support[np.newaxis, :]
        pair_sums = s_i + s_j
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                mask = pair_sums < threshold
                inner = np.sum(w[np.newaxis, :] * w[:, np.newaxis] * mask)
                P += w[k] * w[l] * inner
        
        return P
    
    def neg_P_fixed(weights, support):
        """Negative of P for fixed support."""
        w = np.maximum(weights, 0.0)
        if np.sum(w) < 1e-12:
            return 1e10
        return -compute_P_fixed_support(w, support)
    
    # Try several fixed support configurations with LP-like optimization
    fixed_supports = [
        np.array([0.0, 1.0, 3.0, 9.0]),
        np.array([0.0, 0.5, 2.0, 10.0]),
        np.array([0.0, 1.0, 4.0, 16.0]),
        np.array([0.0, 0.1, 1.0, 10.0]),
        np.array([0.0, 0.5, 2.5, 12.5]),
        np.array([0.0, 1.0, 2.0, 5.0]),
        np.array([0.0, 0.2, 1.0, 8.0]),
        np.array([0.0, 1.0, 3.0, 12.0]),
    ]
    
    for support_fixed in fixed_supports:
        if time.time() > deadline - 20:
            break
        
        try:
            # Use SLSQP to optimize weights on this fixed support
            init_w = np.ones(4) / 4
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds_w = [(0, 1)] * 4
            
            res_w = minimize(
                lambda w: neg_P_fixed(w, support_fixed),
                init_w,
                method='SLSQP',
                bounds=bounds_w,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
            )
            
            if res_w.success:
                P_val = -res_w.fun
                if P_val > best_P:
                    best_P = P_val
                    # Store as [support..., weights...]
                    best_params = np.concatenate([support_fixed, res_w.x])
        except Exception as e:
            pass
    
    # If we found a good solution, use it
    if best_params is not None:
        support = np.maximum(best_params[:4], 0.0)
        weights = np.maximum(best_params[4:], 0.0)
        
        # Ensure validity
        if np.sum(weights) < 1e-9:
            weights = np.ones(4) / 4
            support = np.arange(4, dtype=np.float64)
        
        # Final validation
        support = support.astype(np.float64)
        weights = weights.astype(np.float64)
        
        # Check for NaN or inf
        if np.any(np.isnan(support)) or np.any(np.isinf(support)):
            support = np.arange(4, dtype=np.float64)
            weights = np.ones(4) / 4
        
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            weights = np.ones(4) / 4
        
        # Ensure non-negative
        support = np.maximum(support, 0.0)
        weights = np.maximum(weights, 0.0)
        
        return support, weights
    else:
        # Fallback to baseline
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights


# EVOLVE-BLOCK-END
