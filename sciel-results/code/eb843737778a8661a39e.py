# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Deadline guard
_DEADLINE = time.time() + 600.0
_SAFETY_MARGIN = 30.0  # seconds


def _time_left():
    return _DEADLINE - _SAFETY_MARGIN - time.time()


def _is_valid(support, weights):
    """Check validity constraints."""
    if not isinstance(support, np.ndarray) or not isinstance(weights, np.ndarray):
        return False
    if support.ndim != 1 or weights.ndim != 1:
        return False
    n = len(support)
    if n != len(weights) or n < 1:
        return False
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return False
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return False
    if np.any(support < -1e-9):
        return False
    if np.any(weights < -1e-9):
        return False
    if np.sum(weights) <= 1e-9:
        return False
    return True


def _compute_P_fast(support, weights):
    """
    Fast computation of P[X1+X2+X3 < 2*X4] using sorted pairwise sums.
    """
    n = len(support)
    w = weights / np.sum(weights)
    
    S = support[:, None] + support[None, :]
    W = w[:, None] * w[None, :]
    
    S_flat = S.ravel()
    W_flat = W.ravel()
    
    order = np.argsort(S_flat, kind='mergesort')
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cumsum = np.zeros_like(W_sorted)
    W_cumsum[0] = W_sorted[0]
    W_cumsum[1:] = np.cumsum(W_sorted[1:])
    
    # Build all thresholds: for each (k,l), threshold = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]
    thresh_flat = thresholds.ravel()
    
    # Vectorized searchsorted
    idx = np.searchsorted(S_sorted, thresh_flat, side='left')
    
    # Get cumulative sum values
    inner_vals = np.zeros_like(idx, dtype=np.float64)
    mask = idx > 0
    inner_vals[mask] = W_cumsum[idx[mask] - 1]
    
    # P = sum over k,l of w_k * w_l * inner_vals[k,l]
    w_outer = w[:, None] * w[None, :]
    P = np.sum(w_outer * inner_vals.reshape(n, n))
    
    return float(P)


def _neg_P_slsqp(params):
    """Negative P for SLSQP optimization. Uses log-barrier for positivity."""
    n = len(params) // 2
    support = params[:n]
    weights = params[n:]
    
    # Penalty for invalid values
    if np.any(support < 0) or np.any(weights < 0):
        return 1.0 + np.sum(np.minimum(support, 0)**2) + np.sum(np.minimum(weights, 0)**2)
    
    if np.sum(weights) <= 1e-9:
        return 1.0
    
    try:
        return -_compute_P_fast(support, weights)
    except Exception:
        return 1.0


def _local_refine_slsqp(support, weights, max_iter=200, tol=1e-10):
    """Local refinement using SLSQP with tighter tolerances."""
    n = len(support)
    params = np.concatenate([support, weights])
    
    best_P = _compute_P_fast(support, weights)
    best_params = params.copy()
    
    bounds = [(0.0, 20.0)] * n + [(1e-8, 1.0)] * n
    
    # Try multiple starting points
    starts = [params.copy()]
    for _ in range(5):
        p = params.copy()
        p[:n] += np.random.randn(n) * 0.5
        p[n:] += np.random.randn(n) * 0.1
        p[:n] = np.maximum(0, p[:n])
        p[n:] = np.maximum(1e-8, p[n:])
        starts.append(p)
    
    for p0 in starts:
        if _time_left() < 15:
            break
        
        try:
            result = minimize(
                _neg_P_slsqp, p0, method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'disp': False}
            )
            if result.fun < -best_P:
                best_P = -result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if _time_left() < 15:
        return support, weights
    
    s_opt = best_params[:n]
    w_opt = best_params[n:]
    
    # Ensure validity
    s_opt = np.maximum(0, s_opt)
    w_opt = np.maximum(1e-8, w_opt)
    
    if _is_valid(s_opt, w_opt):
        return s_opt, w_opt
    return support, weights


def _local_refine_lbfgs(support, weights, max_iter=100, tol=1e-8):
    """Local refinement using L-BFGS-B."""
    n = len(support)
    params = np.concatenate([support, weights])
    
    best_P = _compute_P_fast(support, weights)
    best_params = params.copy()
    
    bounds = [(0.0, 20.0)] * n + [(1e-8, 1.0)] * n
    
    for attempt in range(3):
        if _time_left() < 10:
            break
        
        if attempt == 0:
            p0 = params.copy()
        else:
            p0 = params + np.random.randn(len(params)) * 0.1
            p0[:n] = np.maximum(0, p0[:n])
            p0[n:] = np.maximum(1e-8, p0[n:])
        
        try:
            result = minimize(
                _neg_P_slsqp, p0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
            )
            if result.fun < -best_P:
                best_P = -result.fun
                best_params = result.x
        except Exception:
            pass
    
    if _time_left() < 10:
        return support, weights
    
    s_opt = best_params[:n]
    w_opt = best_params[n:]
    
    s_opt = np.maximum(0, s_opt)
    w_opt = np.maximum(1e-8, w_opt)
    
    if _is_valid(s_opt, w_opt):
        return s_opt, w_opt
    return support, weights


def _generate_initial_configurations():
    """Generate diverse initial configurations for optimization."""
    configs = []
    
    # Configuration 1: Geometric spacing
    n1 = 8
    s1 = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    w1 = np.array([0.3, 0.25, 0.2, 0.1, 0.08, 0.04, 0.02, 0.01])
    configs.append((s1, w1))
    
    # Configuration 2: Two-point concentration
    n2 = 6
    s2 = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    w2 = np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
    configs.append((s2, w2))
    
    # Configuration 3: Uniform on [0, 1]
    n3 = 10
    s3 = np.linspace(0, 1, n3)
    w3 = np.ones(n3) / n3
    configs.append((s3, w3))
    
    # Configuration 4: Skewed distribution
    n4 = 8
    s4 = np.array([0.0, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0, 12.0])
    w4 = np.array([0.35, 0.2, 0.15, 0.12, 0.1, 0.05, 0.02, 0.01])
    configs.append((s4, w4))
    
    # Configuration 5: Sparse large values
    n5 = 6
    s5 = np.array([0.0, 1.0, 5.0, 15.0, 45.0, 135.0])
    w5 = np.array([0.5, 0.2, 0.15, 0.1, 0.04, 0.01])
    configs.append((s5, w5))
    
    return configs


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    best_P = -np.inf
    best_support = None
    best_weights = None
    
    # Try diverse initial configurations
    configs = _generate_initial_configurations()
    
    for s_init, w_init in configs:
        if _time_left() < 30:
            break
        
        P = _compute_P_fast(s_init, w_init)
        if P > best_P:
            best_P = P
            best_support = s_init.copy()
            best_weights = w_init.copy()
    
    # Also try uniform distributions with different n
    for n in [5, 6, 7, 8, 9, 10, 12]:
        if _time_left() < 30:
            break
        
        s_uniform = np.linspace(0, 1, n)
        w_uniform = np.ones(n) / n
        P = _compute_P_fast(s_uniform, w_uniform)
        if P > best_P:
            best_P = P
            best_support = s_uniform.copy()
            best_weights = w_uniform.copy()
    
    # Differential evolution global search
    if _time_left() > 120:
        n_de = 8
        bounds_de = [(0.0, 10.0)] * n_de + [(1e-6, 1.0)] * n_de
        
        def _neg_P_de(params):
            n_half = len(params) // 2
            s = params[:n_half]
            w = params[n_half:]
            if np.sum(w) < 1e-6:
                return 1.0
            try:
                return -_compute_P_fast(s, w)
            except Exception:
                return 1.0
        
        try:
            de_result = differential_evolution(
                _neg_P_de,
                bounds=bounds_de,
                maxiter=80,
                popsize=12,
                tol=1e-4,
                seed=42,
                polish=True,
                init='latinhypercube',
                maxfevals=int(min(3000, max(500, _time_left() / 3.0)))
            )
            if de_result.fun < -best_P:
                best_P = -de_result.fun
                best_support = de_result.x[:n_de].copy()
                best_weights = de_result.x[n_de:].copy()
        except Exception:
            pass
    
    # SLSQP local refinement (primary)
    if _time_left() > 40:
        best_support, best_weights = _local_refine_slsqp(best_support, best_weights)
        best_P = _compute_P_fast(best_support, best_weights)
    
    # L-BFGS-B local refinement (secondary)
    if _time_left() > 25:
        best_support, best_weights = _local_refine_lbfgs(best_support, best_weights)
        best_P = _compute_P_fast(best_support, best_weights)
    
    # Final high-precision polish
    if _time_left() > 15:
        n_final = len(best_support)
        params_final = np.concatenate([best_support, best_weights])
        bounds_final = [(0.0, 20.0)] * n_final + [(1e-10, 1.0)] * n_final
        
        try:
            result = minimize(
                _neg_P_slsqp, params_final, method='SLSQP',
                bounds=bounds_final,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            if result.fun < -best_P:
                s_polish = np.maximum(0, result.x[:n_final])
                w_polish = np.maximum(1e-10, result.x[n_final:])
                if _is_valid(s_polish, w_polish):
                    best_support = s_polish
                    best_weights = w_polish
                    best_P = _compute_P_fast(best_support, best_weights)
        except Exception:
            pass
    
    # Final validity check
    if not _is_valid(best_support, best_weights):
        # Fallback to a known valid configuration
        best_support = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
        best_weights = np.array([0.3, 0.25, 0.2, 0.1, 0.08, 0.04, 0.02, 0.01])
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
