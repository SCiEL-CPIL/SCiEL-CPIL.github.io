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

_k = 10
_N_COEFFS = 2 * _k + 1
_N_POINTS = _N_GL * _N_PHI

# Flatten theta and phi grids
_theta_grid, _phi_grid_2d = np.meshgrid(_theta_gl, _phi_grid, indexing='ij')
_phi_flat = _phi_grid_2d.flatten()
_cos_theta_flat = _x_gl[:, None] * np.ones((1, _N_PHI))
_cos_theta_flat = _cos_theta_flat.flatten()

# Weights
_W_flat = np.tile(_w_gl, _N_PHI) * _dphi

# Y matrix: (N_POINTS, _N_COEFFS)
_Y_matrix = np.zeros((_N_POINTS, _N_COEFFS))

for idx, m in enumerate(range(-_k, _k + 1)):
    abs_m = abs(m)
    P = lpmv(abs_m, _k, _x_gl)
    P_flat = np.tile(P, _N_PHI)
    
    n_fact = factorial(_k - abs_m, exact=True)
    d_fact = factorial(_k + abs_m, exact=True)
    N_m = np.sqrt((2 * _k + 1) / (4 * np.pi) * (n_fact / d_fact))
    
    if m == 0:
        _Y_matrix[:, idx] = N_m * P_flat
    elif m > 0:
        cos_term = np.cos(m * _phi_flat)
        _Y_matrix[:, idx] = np.sqrt(2) * N_m * P_flat * cos_term
    else:
        sin_term = np.sin(abs_m * _phi_flat)
        _Y_matrix[:, idx] = np.sqrt(2) * N_m * P_flat * sin_term


def _compute_ratio(c):
    """Compute R = ||f||_4 / ||f||_2 using precomputed quadrature."""
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.inf
    
    f_val = _Y_matrix @ c
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.dot(f2, _W_flat)
    norm4_4 = np.dot(f4, _W_flat)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _compute_ratio_and_grad(c):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient.
    Uses the identity: R = (I4 / I2)^(1/2) where I2 = ∫f², I4 = ∫f⁴.
    dR/dc = (1/(2R)) * (2f³ * I2^(-1/2) * I4^(-3/4) * 4 - 2f * I4^(1/4) * I2^(-3/2) * 2) ... 
    Let me derive properly.
    
    R = I4^(1/4) / I2^(1/2)
    ln R = (1/4) ln I4 - (1/2) ln I2
    d(ln R)/dc = (1/4)(dI4/I4) - (1/2)(dI2/I2)
    dI2/dc = 2 ∫ f Y = 2 Y^T (w * f)
    dI4/dc = 4 ∫ f³ Y = 4 Y^T (w * f³)
    dR/dc = R * [ (1/4)(dI4/I4) - (1/2)(dI2/I2) ]
    """
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.inf, np.zeros_like(c)
    
    f_val = _Y_matrix @ c
    f2 = f_val ** 2
    f3 = f2 * f_val
    f4 = f2 * f2
    
    norm2_sq = np.dot(f2, _W_flat)
    norm4_4 = np.dot(f4, _W_flat)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.inf, np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    R = norm4 / norm2
    
    # Gradient computation
    # dI2/dc = 2 * Y^T (W_flat * f)
    dI2_dc = 2.0 * (_Y_matrix.T @ (_W_flat * f_val))
    # dI4/dc = 4 * Y^T (W_flat * f^3)
    dI4_dc = 4.0 * (_Y_matrix.T @ (_W_flat * f3))
    
    # dR/dc = R * [ (1/4) * dI4_dc / I4 - (1/2) * dI2_dc / I2 ]
    grad = R * (0.25 * dI4_dc / norm4_4 - 0.5 * dI2_dc / norm2_sq)
    
    return R, grad


def _objective(c):
    """Objective function for optimizer (returns scalar)."""
    val, _ = _compute_ratio_and_grad(c)
    return val


def _objective_with_grad(c):
    """Objective and gradient for L-BFGS-B."""
    val, grad = _compute_ratio_and_grad(c)
    return val, grad


def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    that MINIMIZE the norm ratio ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = 620  # seconds, safety margin before 630s kill
    
    best_coeffs = None
    best_ratio = np.inf
    
    def _update_best(c, r):
        nonlocal best_coeffs, best_ratio
        if r < best_ratio and np.all(np.isfinite(c)):
            best_ratio = r
            best_coeffs = c.copy()
    
    n_coeffs = _N_COEFFS
    rng = np.random.default_rng(42)
    
    # 1. Initial structured guesses
    candidates = []
    
    # All ones
    candidates.append(np.ones(n_coeffs))
    
    # Single modes
    for i in range(n_coeffs):
        c = np.zeros(n_coeffs)
        c[i] = 1.0
        candidates.append(c)
    
    # Pairs of modes
    for i in range(n_coeffs):
        for j in range(i + 1, n_coeffs):
            c = np.zeros(n_coeffs)
            c[i] = 1.0
            c[j] = 1.0
            candidates.append(c)
    
    # Sparse random patterns
    for _ in range(40):
        n_nonzero = rng.integers(3, 10)
        indices = rng.choice(n_coeffs, size=n_nonzero, replace=False)
        c = np.zeros(n_coeffs)
        c[indices] = rng.normal(size=n_nonzero)
        candidates.append(c)
        
    # Symmetric patterns
    c_even = np.zeros(n_coeffs)
    c_even[::2] = 1.0
    candidates.append(c_even)
    
    c_odd = np.zeros(n_coeffs)
    c_odd[1::2] = 1.0
    candidates.append(c_odd)
    
    # Alternating signs
    c_alt = np.ones(n_coeffs)
    c_alt[1::2] = -1.0
    candidates.append(c_alt)
    
    # Higher-order harmonic-like patterns
    for freq in [1, 2, 3]:
        c_harm = np.zeros(n_coeffs)
        for i in range(n_coeffs):
            m = i - 10
            c_harm[i] = np.cos(freq * m * np.pi / 10)
        candidates.append(c_harm)
    
    # 2. Evaluate all candidates
    for c in candidates:
        r = _compute_ratio(c)
        _update_best(c, r)
        
    # 3. Local optimization from best candidates using analytic gradients
    scored_candidates = []
    for c in candidates:
        r = _compute_ratio(c)
        scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 20 and run L-BFGS-B with analytic gradient
    for i, (r_init, c_init) in enumerate(scored_candidates[:20]):
        if time.time() > deadline - 120:
            break
        
        try:
            res = minimize(
                _objective_with_grad,
                c_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            c_opt = res.x
            r_opt = _compute_ratio(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # 4. Random restarts with optimization using analytic gradients
    n_restarts = 100
    for i in range(n_restarts):
        if time.time() > deadline - 90:
            break
            
        c_start = rng.normal(size=n_coeffs)
        norm_c = np.linalg.norm(c_start)
        if norm_c < 1e-15:
            continue
        c_start_norm = c_start / norm_c
        
        try:
            res = minimize(
                _objective_with_grad,
                c_start_norm,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            c_opt = res.x
            r_opt = _compute_ratio(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
            
    # 5. Final polish: run L-BFGS-B on the best result with very tight tolerances
    if best_coeffs is not None:
        try:
            res = minimize(
                _objective_with_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-15}
            )
            c_opt = res.x
            r_opt = _compute_ratio(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # 6. Second polish with even tighter settings
    if best_coeffs is not None and time.time() < deadline - 30:
        try:
            res = minimize(
                _objective_with_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            c_opt = res.x
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
