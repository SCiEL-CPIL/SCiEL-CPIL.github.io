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


def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] exactly in float64.
    P = sum over k,l of w_k * w_l * (sum of w_i * w_j over pairs (i,j) with s_i+s_j < 2*s_l - s_k)
    
    We can rewrite: for each (k,l), the inner sum is over all (i,j) where s_i + s_j < 2*s_l - s_k.
    Let's precompute all pairwise sums S[i,j] = s_i + s_j and their weight products W[i,j] = w_i * w_j.
    Then for each (k,l), we need sum of W[i,j] over all (i,j) where S[i,j] < 2*s_l - s_k.
    
    For efficiency with moderate n, we can sort the pairwise sums and use cumulative sums.
    """
    n = len(support)
    w = weights / np.sum(weights)
    
    # Build pairwise sums and weight products
    # S has shape (n, n), S[i,j] = support[i] + support[j]
    S = support[:, None] + support[None, :]
    W = w[:, None] * w[None, :]
    
    # Flatten for sorting
    S_flat = S.ravel()
    W_flat = W.ravel()
    
    # Sort by S_flat
    order = np.argsort(S_flat, kind='mergesort')
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cumsum = np.cumsum(W_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            # Find all indices where S_sorted < threshold (strict inequality)
            # np.searchsorted gives the insertion point
            idx = np.searchsorted(S_sorted, threshold, side='left')
            if idx > 0:
                inner = W_cumsum[idx - 1]
            else:
                inner = 0.0
            P += w[k] * w[l] * inner
    
    return P


def _compute_P_fast(support, weights):
    """
    Faster computation using the fact that we can vectorize the outer (k,l) loop.
    For each (k,l), threshold = 2*s_l - s_k.
    We need sum of W[i,j] where S[i,j] < threshold.
    
    Using sorted S and cumulative W, we can do a vectorized searchsorted.
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
    # Shape (n, n): thresholds[k,l] = 2*s_l - s_k
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


def _neg_P(params):
    """Negative of P for minimization."""
    n = len(params) // 2
    support = params[:n]
    weights = params[n:]
    
    # Ensure support >= 0 and weights >= 0
    if np.any(support < 0) or np.any(weights < 0):
        return 1.0  # Large penalty
    
    if np.sum(weights) <= 1e-9:
        return 1.0
    
    try:
        return -_compute_P_fast(support, weights)
    except Exception:
        return 1.0


def _local_refine(support, weights, max_iter=50, tol=1e-8):
    """Local refinement using Nelder-Mead or L-BFGS-B on the parameters."""
    n = len(support)
    params = np.concatenate([support, weights])
    
    best_P = _compute_P_fast(support, weights)
    best_params = params.copy()
    
    # Use L-BFGS-B with bounds
    bounds = [(0.0, 10.0)] * n + [(1e-6, 1.0)] * n
    
    # Try a few random perturbations and then local optimize
    for attempt in range(3):
        if _time_left() < 10:
            break
        
        if attempt == 0:
            p0 = params.copy()
        else:
            p0 = params + np.random.randn(len(params)) * 0.1
            p0[:n] = np.maximum(0, p0[:n])
            p0[n:] = np.maximum(1e-6, p0[n:])
        
        try:
            result = minimize(
                _neg_P, p0, method='L-BFGS-B',
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
    
    # Ensure validity
    s_opt = np.maximum(0, s_opt)
    w_opt = np.maximum(1e-6, w_opt)
    
    if _is_valid(s_opt, w_opt):
        return s_opt, w_opt
    return support, weights


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    # Start with baseline
    n_start = 8
    support0 = np.arange(n_start, dtype=np.float64)
    weights0 = np.ones(n_start, dtype=np.float64) / n_start
    
    best_P = _compute_P_fast(support0, weights0)
    best_support = support0.copy()
    best_weights = weights0.copy()
    
    # Try a few different sizes
    for n in [6, 8, 10, 12]:
        if _time_left() < 60:
            break
        
        # Initial guess: uniform on [0, 1] or [0, n-1]
        support = np.linspace(0, n - 1, n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        
        P = _compute_P_fast(support, weights)
        if P > best_P:
            best_P = P
            best_support = support.copy()
            best_weights = weights.copy()
    
    # Differential evolution global search
    if _time_left() > 120:
        n_de = 8
        bounds_de = [(0.0, 5.0)] * n_de + [(0.0, 1.0)] * n_de
        
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
                maxiter=100,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=True,
                init='latinhypercube',
                maxfevals=int(min(5000, max(1000, _time_left() / 2.0)))
            )
            if de_result.fun < -best_P:
                best_P = -de_result.fun
                best_support = de_result.x[:n_de].copy()
                best_weights = de_result.x[n_de:].copy()
        except Exception:
            pass
    
    # Local refinement
    if _time_left() > 30:
        best_support, best_weights = _local_refine(best_support, best_weights)
        best_P = _compute_P_fast(best_support, best_weights)
    
    # Final validity check
    if not _is_valid(best_support, best_weights):
        # Fallback to baseline
        best_support = np.arange(8, dtype=np.float64)
        best_weights = np.ones(8, dtype=np.float64) / 8
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
