# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy:
    1. Use scale invariance: fix s_0 = 1 (or 0) to reduce dimensionality.
    2. Fast P computation using sorted pairwise sums + searchsorted.
    3. Multiple restarts with SLSQP + Nelder-Mead polish.
    4. Grow step: try n+1 points from best n-point solution.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    start_time = time.time()
    deadline = start_time + 570  # 570s safety margin before 630s kill
    
    def _compute_P(support, weights):
        """
        Compute P[X1+X2+X3 < 2*X4] efficiently using sorted pairwise sums.
        
        P = sum_{k,l} w_k w_l * Prob[X1+X2 < 2*s_l - s_k]
        
        For fixed support, precompute all n^2 pairwise sums sorted, then for each
        threshold use searchsorted to find the count, and compute weighted sum.
        """
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 0.0
        w = weights / w_sum
        
        # Compute all pairwise sums s_i + s_j (n x n matrix)
        pair_sums = support[:, np.newaxis] + support[np.newaxis, :]  # (n, n)
        
        # Flatten and sort for efficient threshold queries
        flat_sums = pair_sums.ravel()  # (n^2,)
        flat_w = (w[:, np.newaxis] * w[np.newaxis, :]).ravel()  # (n^2,)
        
        # Sort by sum value
        sort_idx = np.argsort(flat_sums, kind='stable')
        sorted_sums = flat_sums[sort_idx]
        sorted_w = flat_w[sort_idx]
        # Cumulative sum of weights for prefix sums
        cum_w = np.cumsum(sorted_w)
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                # Find index of first element >= threshold (strict inequality)
                idx = np.searchsorted(sorted_sums, threshold, side='left')
                inner = cum_w[idx - 1] if idx > 0 else 0.0
                P += w[k] * w[l] * inner
        
        return P
    
    def _neg_P_from_params(params, n):
        """
        Negative of P for minimization.
        params layout: [s_1, s_2, ..., s_{n-1}, w_0, w_1, ..., w_{n-1}]
        s_0 is fixed at 1.0 (scale invariance).
        """
        # Support: s_0 = 1.0, rest from params
        support = np.empty(n)
        support[0] = 1.0
        if n > 1:
            support[1:] = np.maximum(params[:n-1], 1e-15)
        
        # Weights
        weights = np.maximum(params[n-1:], 0.0)
        w_sum = np.sum(weights)
        if w_sum < 1e-12:
            return 1e10
        
        return -_compute_P(support, weights)
    
    def _optimize_n(n, max_time_per_restart=15.0, max_restarts=12):
        """
        Optimize for a given number of support points n.
        Returns (support, weights, P_value).
        """
        best_P = -1.0
        best_support = None
        best_weights = None
        
        # Generate diverse initial conditions
        rng = np.random.default_rng(42 + n)
        
        def make_init():
            """Create a random initial condition for n points."""
            if n == 1:
                return np.array([1.0])
            
            # Support points: s_0 = 1 fixed, others random in [0.1, 100]
            s = np.sort(rng.uniform(0.1, 100.0, size=n-1))
            # Weights: random Dirichlet
            w = rng.dirichlet(np.ones(n))
            
            return np.concatenate([s, w])
        
        bounds = [(1e-15, None)] * (n-1) + [(0, 1)] * n
        
        # Phase 1: SLSQP from multiple restarts
        for restart in range(max_restarts):
            if time.time() > deadline - 30:
                break
            
            init = make_init()
            
            try:
                res = minimize(
                    lambda p: _neg_P_from_params(p, n),
                    init,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False}
                )
                
                if res.fun < -best_P or best_P < 0:
                    P_val = -res.fun
                    if P_val > best_P:
                        best_P = P_val
                        support = np.empty(n)
                        support[0] = 1.0
                        if n > 1:
                            support[1:] = np.maximum(res.x[:n-1], 1e-15)
                        weights = np.maximum(res.x[n-1:], 0.0)
                        if np.sum(weights) > 1e-12:
                            best_support = support.copy()
                            best_weights = weights.copy()
            except Exception:
                pass
        
        # Phase 2: Nelder-Mead polish
        if best_support is not None and time.time() < deadline - 20:
            try:
                init_polish = np.concatenate([best_support[1:], best_weights])
                res_polish = minimize(
                    lambda p: _neg_P_from_params(p, n),
                    init_polish,
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                
                P_val = -res_polish.fun
                if P_val > best_P:
                    best_P = P_val
                    support = np.empty(n)
                    support[0] = 1.0
                    if n > 1:
                        support[1:] = np.maximum(res_polish.x[:n-1], 1e-15)
                    weights = np.maximum(res_polish.x[n-1:], 0.0)
                    if np.sum(weights) > 1e-12:
                        best_support = support.copy()
                        best_weights = weights.copy()
            except Exception:
                pass
        
        if best_support is not None:
            return best_support, best_weights, best_P
        return None, None, -1.0
    
    # Try n = 2, 3, 4, 5 points
    best_overall_P = -1.0
    best_overall_support = None
    best_overall_weights = None
    
    for n in [2, 3, 4, 5]:
        if time.time() > deadline - 60:
            break
        
        support, weights, P_val = _optimize_n(n)
        
        if support is not None and P_val > best_overall_P:
            best_overall_P = P_val
            best_overall_support = support
            best_overall_weights = weights
        
        print(f"n={n}: P={P_val:.6f}", file=sys.stderr)
    
    # Grow step: try n+1 from best n-point solution
    if best_overall_support is not None and time.time() < deadline - 40:
        n_best = len(best_overall_support)
        n_grow = n_best + 1
        if n_grow <= 6:
            # Initialize by adding a new point
            try:
                rng = np.random.default_rng(123)
                # Try several placements for the new point
                for trial in range(3):
                    if time.time() > deadline - 30:
                        break
                    
                    new_s = rng.uniform(0.01, 50.0)
                    new_w = rng.uniform(0.01, 0.3)
                    
                    # Build initial condition
                    init_s = np.concatenate([best_overall_support[1:], [new_s]])
                    init_s = np.sort(init_s)
                    init_w = np.concatenate([best_overall_weights, [new_w]])
                    init_w = init_w / np.sum(init_w)
                    
                    init = np.concatenate([init_s, init_w])
                    
                    bounds = [(1e-15, None)] * (n_grow-1) + [(0, 1)] * n_grow
                    
                    res = minimize(
                        lambda p: _neg_P_from_params(p, n_grow),
                        init,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
                    )
                    
                    P_val = -res.fun
                    if P_val > best_overall_P:
                        support = np.empty(n_grow)
                        support[0] = 1.0
                        support[1:] = np.maximum(res.x[:n_grow-1], 1e-15)
                        weights = np.maximum(res.x[n_grow-1:], 0.0)
                        if np.sum(weights) > 1e-12:
                            best_overall_P = P_val
                            best_overall_support = support.copy()
                            best_overall_weights = weights.copy()
                            print(f"grow to n={n_grow}: P={P_val:.6f}", file=sys.stderr)
            except Exception:
                pass
    
    # Final validation and return
    if best_overall_support is not None:
        support = np.maximum(best_overall_support, 0.0)
        weights = np.maximum(best_overall_weights, 0.0)
        
        # Ensure validity
        if np.sum(weights) < 1e-9:
            support = np.arange(1, len(support)+1, dtype=np.float64)
            weights = np.ones(len(support), dtype=np.float64) / len(support)
        
        support = support.astype(np.float64)
        weights = weights.astype(np.float64)
        
        # Check for NaN or inf
        if np.any(np.isnan(support)) or np.any(np.isinf(support)):
            support = np.arange(1, 5, dtype=np.float64)
            weights = np.ones(4, dtype=np.float64) / 4
        
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            weights = np.ones(len(support), dtype=np.float64) / len(support)
        
        support = np.maximum(support, 0.0)
        weights = np.maximum(weights, 0.0)
        
        return support, weights
    else:
        # Fallback
        support = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        return support, weights


# EVOLVE-BLOCK-END
