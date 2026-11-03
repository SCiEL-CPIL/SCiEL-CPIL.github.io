import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

def _precompute_conv_matrix():
    """Precompute the autoconvolution kernel for step functions using vectorized ops."""
    dw = 1.0 / N
    x = np.linspace(-0.25, 0.25, N + 1)
    
    n_pts = 8 * N
    t_grid = np.linspace(-0.5, 0.5, n_pts + 1)
    midpoints = (t_grid[:-1] + t_grid[1:]) / 2.0
    
    # Vectorized computation of K[i,j,k]
    # K[i,j,k] = max(0, min(x[i+1], t_k - x[j]) - max(x[i], t_k - x[j+1]))
    # for t_k in [x[i]+x[j], x[i+1]+x[j+1]]
    
    # Create index arrays
    i_idx = np.arange(N)[:, None, None]  # (N, 1, 1)
    j_idx = np.arange(N)[None, :, None]  # (1, N, 1)
    k_idx = np.arange(n_pts)[None, None, :]  # (1, 1, n_pts)
    
    x_i = x[i_idx]  # (N, 1, 1)
    x_i1 = x[i_idx + 1]  # (N, 1, 1)
    x_j = x[j_idx]  # (1, N, 1)
    x_j1 = x[j_idx + 1]  # (1, N, 1)
    t_k = midpoints[k_idx]  # (1, 1, n_pts)
    
    # Compute lo and hi
    lo = np.maximum(x_i, t_k - x_j1)  # (N, N, n_pts)
    hi = np.minimum(x_i1, t_k - x_j)  # (N, N, n_pts)
    
    # Valid range: t in [x_i + x_j, x_i1 + x_j1]
    t_lo = x_i + x_j  # (N, N, 1)
    t_hi = x_i1 + x_j1  # (N, N, 1)
    
    valid = (t_k >= t_lo) & (t_k <= t_hi)  # (N, N, n_pts)
    
    K = np.where(valid, np.maximum(0.0, hi - lo), 0.0)  # (N, N, n_pts)
    
    return K, midpoints

_K, _t_mid = _precompute_conv_matrix()
_K_flat = _K.reshape(N * N, -1)

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
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ _K_flat
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)


def _score_neg(h):
    """Negative score for minimization."""
    return -_compute_score(h)

def _epigraph_objective(z):
    """
    Epigraph formulation: minimize -L2_sq / (L1 * t) subject to |autoconv_k| <= t.
    Variables: z = [h_0, ..., h_{N-1}, t]
    We minimize -L2_sq / (L1 * t) with penalty for constraint violations.
    """
    h = z[:N]
    t = z[N]
    
    h = np.maximum(h, 0.0)
    if np.sum(h) < 1e-15 or t < 1e-15:
        return 1e10
    
    dw = 1.0 / N
    L1 = np.sum(h) * dw
    L2_sq = np.sum(h**2) * dw
    
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ _K_flat
    abs_ac = np.abs(autoconv)
    
    # Penalty for constraint violation: sum of max(0, |ac_k| - t)^2
    violations = np.maximum(0.0, abs_ac - t)
    penalty = np.sum(violations**2)
    
    # Objective: -L2_sq / (L1 * t) + penalty
    obj = -L2_sq / (L1 * t) + 1e4 * penalty
    
    return obj

def _epigraph_grad(z):
    """Gradient for the epigraph formulation."""
    h = z[:N]
    t = z[N]
    
    h_clipped = np.maximum(h, 0.0)
    if np.sum(h_clipped) < 1e-15 or t < 1e-15:
        return np.zeros(N + 1)
    
    dw = 1.0 / N
    L1 = np.sum(h_clipped) * dw
    L2_sq = np.sum(h_clipped**2) * dw
    
    h_outer = np.outer(h_clipped, h_clipped).flatten()
    autoconv = h_outer @ _K_flat
    abs_ac = np.abs(autoconv)
    
    # Gradient of -L2_sq / (L1 * t) w.r.t. h_i
    # d(-L2_sq / (L1 * t))/dh_i = -[2*h_i*dw*(L1*t) - L2_sq*dw*t] / (L1*t)^2
    # = -[2*h_i*L1*t - L2_sq*t] / (L1^2 * t)
    # = t*(L2_sq - 2*h_i*L1) / (L1^2 * t) ... let me redo this
    
    # f = -L2_sq / (L1 * t)
    # df/dh_i = -[2*h_i*dw * L1*t - L2_sq * dw * t] / (L1*t)^2
    #          = -dw * t * (2*h_i*L1 - L2_sq) / (L1^2 * t^2)
    #          = -dw * (2*h_i*L1 - L2_sq) / (L1^2 * t)
    
    grad_h = np.zeros(N)
    for i in range(N):
        if h_clipped[i] > 0:
            grad_h[i] = -dw * (2*h_clipped[i]*L1 - L2_sq) / (L1**2 * t)
    
    # Gradient w.r.t. t: df/dt = L2_sq / (L1 * t^2)
    grad_t = L2_sq / (L1 * t**2)
    
    # Gradient of penalty: 2 * 1e4 * sum over k of max(0, |ac_k| - t) * d|ac_k|/dh_i
    # This is complex; use numerical gradient for the penalty part
    
    eps = 1e-8
    grad = np.zeros(N + 1)
    for i in range(N):
        if h_clipped[i] > eps:
            z_plus = z.copy()
            z_plus[i] += eps
            z_minus = z.copy()
            z_minus[i] -= eps
            grad[i] = (_epigraph_objective(z_plus) - _epigraph_objective(z_minus)) / (2 * eps)
        else:
            z_plus = z.copy()
            z_plus[i] += eps
            grad[i] = (_epigraph_objective(z_plus) - _epigraph_objective(z)) / eps
    
    # Gradient w.r.t. t
    z_plus = z.copy()
    z_plus[N] += eps
    z_minus = z.copy()
    z_minus[N] -= eps
    if z_minus[N] < 1e-10:
        z_minus[N] = 1e-10
    grad[N] = (_epigraph_objective(z_plus) - _epigraph_objective(z_minus)) / (2 * eps)
    
    return grad


def _subgradient_step(h, lr=0.01, iterations=100):
    """Subgradient descent step for the non-smooth objective."""
    h = np.asarray(h, dtype=float).copy()
    h = np.maximum(h, 0.0)
    best_h = h.copy()
    best_score = _compute_score(h)
    
    for it in range(iterations):
        if np.sum(h) < 1e-15:
            break
        
        dw = 1.0 / N
        L1 = np.sum(h) * dw
        L2_sq = np.sum(h**2) * dw
        
        h_outer = np.outer(h, h).flatten()
        autoconv = h_outer @ _K_flat
        abs_ac = np.abs(autoconv)
        sup_conv = np.max(abs_ac)
        
        if sup_conv < 1e-15:
            break
        
        score = L2_sq / (L1 * sup_conv)
        
        # Find the index where supremum is achieved
        sup_idx = np.argmax(abs_ac)
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] = max(0, h[i] + eps)
            h_minus = h.copy()
            h_minus[i] = max(0, h[i] - eps)
            
            score_plus = _compute_score(h_plus)
            score_minus = _compute_score(h_minus)
            
            if h[i] > eps:
                grad[i] = (score_plus - score_minus) / (2 * eps)
            else:
                grad[i] = (score_plus - score_minus) / eps
        
        # Gradient ascent (we want to maximize score)
        h = h + lr * grad
        h = np.maximum(h, 0.0)
        
        # Normalize to keep scale reasonable
        if np.sum(h) > 1e10:
            h = h / np.sum(h) * N
        
        current_score = _compute_score(h)
        if current_score > best_score:
            best_h = h.copy()
            best_score = current_score
    
    return best_h, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start from known good solution
    h0 = np.zeros(N)
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = h0.copy()
    best_score = _compute_score(best_h)
    
    # Phase 1: Direct L-BFGS-B with exact objective from multiple starts
    starts = [
        h0.copy(),
        np.array([1.0/3.0, 1.0/3.0, 1.0/3.0] + [0.0]*(N-3)),
        np.array([0.5, 0.5] + [0.0]*(N-2)),
        np.array([1.0] + [0.0]*(N-1)),
        np.array([0.8, 0.2] + [0.0]*(N-2)),
        np.array([0.6, 0.3, 0.1] + [0.0]*(N-3)),
        np.ones(N) / N,
        np.array([1.0/N]*N),
    ]
    
    for s in starts:
        if time.time() > deadline - 60:
            break
        try:
            result = minimize(_score_neg, s, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-10})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 2: Epigraph formulation optimization
    if time.time() < deadline - 30:
        try:
            # Initialize t as current sup_conv
            h_init = best_h.copy()
            h_outer = np.outer(h_init, h_init).flatten()
            autoconv = h_outer @ _K_flat
            t_init = np.max(np.abs(autoconv)) + 1e-6
            z_init = np.concatenate([h_init, [t_init]])
            
            result = minimize(_epigraph_objective, z_init, method='L-BFGS-B',
                             jac=_epigraph_grad,
                             bounds=[(0, 10)] * N + [(1e-10, 100)],
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            z_trial = result.x
            h_trial = np.maximum(z_trial[:N], 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 3: Subgradient refinement
    if time.time() < deadline - 15:
        try:
            h_sub, score_sub = _subgradient_step(best_h, lr=0.05, iterations=50)
            if score_sub > best_score:
                best_h = h_sub
                best_score = score_sub
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            result = minimize(_score_neg, best_h, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14,
                                     'initial_simplex_size': 1e-5})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 6: Another subgradient pass with smaller lr
    if time.time() < deadline - 2:
        try:
            h_sub, score_sub = _subgradient_step(best_h, lr=0.01, iterations=30)
            if score_sub > best_score:
                best_h = h_sub
                best_score = score_sub
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
