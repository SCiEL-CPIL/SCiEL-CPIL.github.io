# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Precompute quadrature nodes and weights
_N_GL = 70
_N_PHI = 130
_x_gl, _w_gl = np.polynomial.legendre.leggauss(_N_GL)
_theta_gl = np.arccos(_x_gl)
_phi_grid = 2 * np.pi * np.arange(_N_PHI) / _N_PHI
_dphi = 2 * np.pi / _N_PHI

# Precompute spherical harmonic values at all quadrature points
# Shape: (21, 70, 130) for m = -10..10
_k = 10
_N_COEFFS = 2 * _k + 1
_Y_precomputed = np.zeros((_N_COEFFS, _N_GL, _N_PHI))

for idx, m in enumerate(range(-_k, _k + 1)):
    abs_m = abs(m)
    P = lpmv(abs_m, _k, _x_gl)  # shape (70,)
    
    n_fact = factorial(_k - abs_m, exact=True)
    d_fact = factorial(_k + abs_m, exact=True)
    N_m = np.sqrt((2 * _k + 1) / (4 * np.pi) * (n_fact / d_fact))
    
    if m == 0:
        _Y_precomputed[idx] = N_m * P[:, None]
    elif m > 0:
        _Y_precomputed[idx] = np.sqrt(2) * N_m * P[:, None] * np.cos(m * _phi_grid)[None, :]
    else:
        _Y_precomputed[idx] = np.sqrt(2) * N_m * P[:, None] * np.sin(abs_m * _phi_grid)[None, :]

# Precompute weights for norm calculations
_W_matrix = _w_gl[:, None] * _dphi

def _compute_ratio(coeffs):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed quadrature.
    coeffs: length 21 vector c_m for m=-10..10
    Returns: ratio R (lower is better)
    """
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        return np.inf
    
    f_val = np.tensordot(coeffs, _Y_precomputed, axes=(0, 0))
    
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.sum(f2 * _W_matrix)
    norm4_4 = np.sum(f4 * _W_matrix)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _compute_ratio_and_grad(coeffs):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient with respect to coeffs.
    """
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        return np.inf, np.zeros_like(coeffs)
    
    f_val = np.tensordot(coeffs, _Y_precomputed, axes=(0, 0))
    
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.sum(f2 * _W_matrix)
    norm4_4 = np.sum(f4 * _W_matrix)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    R = norm4 / norm2
    
    weighted_f3 = (f_val ** 3) * _W_matrix
    dN4_dc = np.tensordot(_Y_precomputed, weighted_f3, axes=([1, 2], [0, 1]))
    
    weighted_f = f_val * _W_matrix
    dN2_dc = np.tensordot(_Y_precomputed, weighted_f, axes=([1, 2], [0, 1]))
    
    grad = dN4_dc / (norm2 * norm4**3) - norm4 * dN2_dc / (norm2**3)
    
    return R, grad


def _trust_region_optimize(c0, max_iter=500, tol=1e-12, deadline=None):
    """
    Trust-region method on the unit sphere for minimizing R = ||f||_4/||f||_2.
    Uses exact line search along the trust-region boundary.
    """
    c = c0 / np.linalg.norm(c0)
    R, grad = _compute_ratio_and_grad(c)
    
    # Project gradient to tangent space
    grad_tan = grad - np.dot(grad, c) * c
    
    # Initial trust region radius
    delta = 0.1
    min_delta = 1e-15
    max_delta = 0.5
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad_tan_norm = np.linalg.norm(grad_tan)
        if grad_tan_norm < tol:
            break
        
        # Try to take a full step in the negative gradient direction
        # Project -grad_tan to the sphere
        step_dir = -grad_tan / grad_tan_norm
        
        # Compute the point on the sphere in the direction of step_dir
        # c_new = cos(alpha) * c + sin(alpha) * step_dir
        # where alpha is chosen to stay within trust region
        
        # Try different step sizes
        best_alpha = None
        best_R = R
        best_c = c.copy()
        
        # Try a range of step sizes
        for alpha in np.concatenate([
            np.linspace(0.01, delta, 10),
            np.linspace(delta, min(delta * 2, max_delta), 5)
        ]):
            if np.sin(alpha) > 1:
                continue
            c_trial = np.cos(alpha) * c + np.sin(alpha) * step_dir
            c_trial = c_trial / np.linalg.norm(c_trial)
            
            R_trial, _ = _compute_ratio_and_grad(c_trial)
            if R_trial < best_R:
                best_R = R_trial
                best_alpha = alpha
                best_c = c_trial.copy()
        
        if best_alpha is None:
            # No improvement, shrink trust region
            delta *= 0.5
            if delta < min_delta:
                break
            continue
        
        # Accept step and adjust trust region
        c = best_c
        R = best_R
        
        # Update gradient
        R, grad = _compute_ratio_and_grad(c)
        grad_tan = grad - np.dot(grad, c) * c
        
        # Adjust trust region based on improvement
        if best_R < R - 1e-15:
            delta = min(delta * 1.5, max_delta)
        else:
            delta = max(delta * 0.7, min_delta)
    
    return c, R


def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    that MINIMIZE the norm ratio ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    best_coeffs = None
    best_ratio = np.inf
    
    def _update_best(c, r):
        nonlocal best_coeffs, best_ratio
        if r < best_ratio and np.all(np.isfinite(c)):
            best_ratio = r
            best_coeffs = c.copy()
    
    n_coeffs = _N_COEFFS
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All ones
    c0 = np.ones(n_coeffs)
    candidates.append(c0)
    
    # 2. Single modes
    for m_idx in range(n_coeffs):
        c_single = np.zeros(n_coeffs)
        c_single[m_idx] = 1.0
        candidates.append(c_single)
    
    # 3. Pairs of modes
    for i in range(n_coeffs):
        for j in range(i + 1, n_coeffs):
            c_pair = np.zeros(n_coeffs)
            c_pair[i] = 1.0
            c_pair[j] = 1.0
            candidates.append(c_pair)
    
    # 4. Random normalized vectors
    for _ in range(50):
        c_rand = rng.normal(size=n_coeffs)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # 5. Structured patterns
    # Even m only
    c_even = np.zeros(n_coeffs)
    c_even[::2] = 1.0
    candidates.append(c_even)
    
    # Odd m only
    c_odd = np.zeros(n_coeffs)
    c_odd[1::2] = 1.0
    candidates.append(c_odd)
    
    # Positive m only
    c_pos = np.zeros(n_coeffs)
    c_pos[10:] = 1.0
    candidates.append(c_pos)
    
    # Negative m only
    c_neg = np.zeros(n_coeffs)
    c_neg[:11] = 1.0
    candidates.append(c_neg)
    
    # 6. Sparse patterns with specific structures
    for _ in range(30):
        n_nonzero = rng.integers(3, 9)
        indices = rng.choice(n_coeffs, size=n_nonzero, replace=False)
        c_sparse = np.zeros(n_coeffs)
        c_sparse[indices] = rng.normal(size=n_nonzero)
        c_sparse /= np.linalg.norm(c_sparse)
        candidates.append(c_sparse)
    
    # Sort candidates by their ratio and take top ones for optimization
    scored_candidates = []
    for c in candidates:
        r = _compute_ratio(c)
        if np.isfinite(r):
            scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Update best from initial candidates
    if scored_candidates:
        _update_best(scored_candidates[0][1], scored_candidates[0][0])
    
    # Define objective with gradient for SLSQP
    def _obj_with_grad(c):
        R, grad = _compute_ratio_and_grad(c)
        if np.isinf(R):
            return 1e10, np.zeros_like(c)
        return R, grad
    
    # Equality constraint: ||c||^2 = 1
    def _norm_constraint(c):
        return np.dot(c, c) - 1.0
    
    def _norm_constraint_grad(c):
        return 2.0 * c
    
    constraints = [{'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_constraint_grad}]
    
    # Optimize from top candidates using SLSQP with analytic gradient
    n_slsqp = 20
    for i, (r_init, c_init) in enumerate(scored_candidates[:n_slsqp]):
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(
                _obj_with_grad,
                c_init,
                method='SLSQP',
                jac=True,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            c_opt = res.x
            c_opt = c_opt / np.linalg.norm(c_opt)
            r_opt = _compute_ratio(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # Additional random restarts with SLSQP
    n_extra = 30
    for i in range(n_extra):
        if time.time() > deadline - 40:
            break
        
        c_start = rng.normal(size=n_coeffs)
        norm_c = np.linalg.norm(c_start)
        if norm_c < 1e-15:
            continue
        c_start_norm = c_start / norm_c
        
        try:
            res = minimize(
                _obj_with_grad,
                c_start_norm,
                method='SLSQP',
                jac=True,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            if np.isfinite(res.fun) and res.fun < best_ratio:
                c_opt = res.x
                c_opt = c_opt / np.linalg.norm(c_opt)
                r_opt = _compute_ratio(c_opt)
                if np.isfinite(r_opt):
                    _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # Final polish: refine best solution with tighter tolerances
    if best_coeffs is not None and time.time() < deadline - 20:
        try:
            res = minimize(
                _obj_with_grad,
                best_coeffs / np.linalg.norm(best_coeffs),
                method='SLSQP',
                jac=True,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            c_opt = res.x
            c_opt = c_opt / np.linalg.norm(c_opt)
            r_opt = _compute_ratio(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # Final validation
    if best_coeffs is None or np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(n_coeffs)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            best_coeffs = np.ones(n_coeffs)
        else:
            best_coeffs = best_coeffs / max_abs * 1.0
            if np.max(np.abs(best_coeffs)) < 1e-3:
                best_coeffs = np.ones(n_coeffs)
    
    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
