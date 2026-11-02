import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

# Precompute the autoconvolution kernel for step functions
# For step functions on [-1/4, 1/4] with N intervals of width dw=1/N:
# f(x) = h_i for x in [x_i, x_{i+1}) where x_i = -1/4 + i*dw
# (f*f)(t) = sum over i,j of h_i * h_j * overlap_length(i, j, t)
# where overlap is the measure of s such that s in [x_i, x_{i+1}) and t-s in [x_j, x_{j+1})
# This equals the measure of intersection of [x_i, x_{i+1}) and [t - x_{j+1}, t - x_j]

def _precompute_conv_matrix():
    """Precompute the autoconvolution values at grid points for unit step functions."""
    dw = 1.0 / N
    x = np.linspace(-0.25, 0.25, N + 1)
    
    # Sample autoconvolution at fine grid points
    # Support of f*f is [-1/2, 1/2]
    n_pts = 4 * N  # fine grid
    t_grid = np.linspace(-0.5, 0.5, n_pts + 1)
    midpoints = (t_grid[:-1] + t_grid[1:]) / 2.0
    
    # For each pair (i,j), compute the contribution to autoconv at each t
    # conv(t) = sum_{i,j} h_i h_j * K_{ij}(t)
    # where K_{ij}(t) = measure of {s: s in [x_i,x_{i+1}), t-s in [x_j,x_{j+1})}
    # = measure of intersection of [x_i, x_{i+1}) and [t-x_{j+1}, t-x_j]
    
    # K_{ij}(t) is a piecewise linear function of t
    # For fixed i,j, it's nonzero when [x_i, x_{i+1}) intersects [t-x_{j+1}, t-x_j]
    # i.e., when t is in [x_i + x_j, x_{i+1} + x_{j+1}]
    
    # We'll compute K_{ij} at the midpoint grid
    K = np.zeros((N, N, n_pts))
    for i in range(N):
        for j in range(N):
            # K_{ij}(t) = max(0, min(x_{i+1}, t - x_j) - max(x_i, t - x_{j+1}))
            # for t in [x_i+x_j, x_{i+1}+x_{j+1}]
            t_lo = x[i] + x[j]
            t_hi = x[i+1] + x[j+1]
            for k in range(n_pts):
                t = midpoints[k]
                if t < t_lo or t > t_hi:
                    continue
                lo = max(x[i], t - x[j+1])
                hi = min(x[i+1], t - x[j])
                if lo < hi:
                    K[i, j, k] = hi - lo
    
    return K, midpoints

_K, _t_mid = _precompute_conv_matrix()

def _compute_score(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    dw = 1.0 / N
    
    # L1 norm (integral)
    L1 = np.sum(h) * dw
    
    # L2 norm squared (integral of h^2)
    L2_sq = np.sum(h**2) * dw
    
    # Compute autoconvolution using precomputed kernel
    # autoconv[k] = sum_{i,j} h_i * h_j * K[i,j,k]
    # Use matrix multiplication: autoconv = (h @ K_reshaped)
    K_flat = _K.reshape(N * N, -1)
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ K_flat
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)


def _score_neg(h):
    """Negative score for minimization."""
    return -_compute_score(h)

def _score_neg_smooth(h, beta=100.0):
    """Negative smooth surrogate score using log-sum-exp for the supremum."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    dw = 1.0 / N
    L1 = np.sum(h) * dw
    L2_sq = np.sum(h**2) * dw
    
    K_flat = _K.reshape(N * N, -1)
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ K_flat
    
    # Smooth max using log-sum-exp
    abs_ac = np.abs(autoconv)
    # Soft max: log(sum(exp(beta * x))) / beta approximates max(x)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 0.0
    
    # Shift for numerical stability
    shifted = beta * (abs_ac - max_val)
    log_sum = max_val + np.log(np.sum(np.exp(shifted))) / beta
    
    if log_sum < 1e-15:
        return 0.0
    
    return -(L2_sq / (L1 * log_sum))


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s limit)
    
    # Start from known good solution
    h0 = np.zeros(N)
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = h0.copy()
    best_score = _compute_score(best_h)
    
    # Use smooth surrogate for gradient-based optimization
    # This helps L-BFGS-B navigate the non-smooth landscape
    
    # Phase 1: Smooth surrogate optimization from multiple starts
    starts = [
        h0.copy(),
        np.array([1.0/3.0, 1.0/3.0, 1.0/3.0] + [0.0]*(N-3)),
        np.array([0.5, 0.5] + [0.0]*(N-2)),
        np.array([1.0] + [0.0]*(N-1)),
        np.array([0.8, 0.2] + [0.0]*(N-2)),
        np.array([0.6, 0.3, 0.1] + [0.0]*(N-3)),
    ]
    
    for s in starts:
        if time.time() > deadline - 30:
            break
        try:
            # Use smooth objective with moderate beta
            result = minimize(_score_neg_smooth, s, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             args=(50.0,),
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 2: Increase beta for sharper approximation
    for beta in [100.0, 200.0, 500.0]:
        if time.time() > deadline - 15:
            break
        try:
            result = minimize(_score_neg_smooth, best_h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             args=(beta,),
                             options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-10})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 3: Direct optimization with exact (non-smooth) objective
    if time.time() < deadline - 10:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-10})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish (derivative-free, handles non-smoothness)
    if time.time() < deadline - 5:
        try:
            result = minimize(_score_neg, best_h, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                     'initial_simplex_size': 1e-4})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 2:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Ensure nonnegativity and clip tiny negatives
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
