# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize

_DEADLINE = time.time() + 610

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2.0 * np.pi / 130
    weights_1d = w_gl * dphi
    return cos_theta, phi, weights_1d

def _compute_harmonics_matrix(cos_theta, phi):
    """
    Compute the matrix X where X[i, j] = Y_j(ct_i, phi_j).
    j ranges from 0 to 20, corresponding to m = -10 to 10.
    """
    N_pts_ct = len(cos_theta)
    N_pts_phi = len(phi)
    N_pts = N_pts_ct * N_pts_phi
    X = np.zeros((N_pts, 21))
    
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    ct_flat = ct.flatten()
    ph_flat = ph.flatten()
    
    for k, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + 
                       gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
        N_m = np.exp(log_N)
        
        P = lpmv(abs_m, 10, ct_flat)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * ph_flat)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * ph_flat)
            
        X[:, k] = Y
        
    return X

def _ratio_and_grad(c, X, w_flat):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient.
    """
    f = X @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    
    S2 = np.sum(w_flat * f2)
    S4 = np.sum(w_flat * f4)
    
    if S2 < 1e-10 or S4 < 1e-10:
        return 1e10, np.zeros(21)
        
    N2 = np.sqrt(S2)
    N4 = S4 ** 0.25
    R = N4 / N2
    
    term1 = X.T @ (w_flat * f3)
    term2 = X.T @ (w_flat * f)
    
    grad_log_R = term1 / S4 - term2 / S2
    grad_R = R * grad_log_R
    
    return R, grad_R

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    cos_theta, phi, w_1d = _build_quadrature()
    X = _compute_harmonics_matrix(cos_theta, phi)
    w_flat = np.tile(w_1d, 130)
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    starts = []
    # m=0
    c0 = np.zeros(21)
    c0[10] = 1.0
    starts.append(c0)
    
    # m=10
    c10 = np.zeros(21)
    c10[20] = 1.0
    starts.append(c10)
    
    # m=-10
    c_10 = np.zeros(21)
    c_10[0] = 1.0
    starts.append(c_10)
    
    # m=5, -5
    c5 = np.zeros(21)
    c5[15] = 1.0
    starts.append(c5)
    
    c_5 = np.zeros(21)
    c_5[5] = 1.0
    starts.append(c_5)
    
    # m=4, -4
    c4 = np.zeros(21)
    c4[14] = 1.0
    starts.append(c4)
    
    c_4 = np.zeros(21)
    c_4[6] = 1.0
    starts.append(c_4)
    
    # m=8, -8
    c8 = np.zeros(21)
    c8[18] = 1.0
    starts.append(c8)
    
    c_8 = np.zeros(21)
    c_8[2] = 1.0
    starts.append(c_8)
    
    # m=2, -2
    c2 = np.zeros(21)
    c2[12] = 1.0
    starts.append(c2)
    
    c_2 = np.zeros(21)
    c_2[8] = 1.0
    starts.append(c_2)
    
    # m=6, -6
    c6 = np.zeros(21)
    c6[16] = 1.0
    starts.append(c6)
    
    c_6 = np.zeros(21)
    c_6[4] = 1.0
    starts.append(c_6)
    
    # m=3, -3
    c3 = np.zeros(21)
    c3[13] = 1.0
    starts.append(c3)
    
    c_3 = np.zeros(21)
    c_3[7] = 1.0
    starts.append(c_3)
    
    # m=7, -7
    c7 = np.zeros(21)
    c7[17] = 1.0
    starts.append(c7)
    
    c_7 = np.zeros(21)
    c_7[3] = 1.0
    starts.append(c_7)
    
    # m=9, -9
    c9 = np.zeros(21)
    c9[19] = 1.0
    starts.append(c9)
    
    c_9 = np.zeros(21)
    c_9[1] = 1.0
    starts.append(c_9)
    
    # Structured mixed starts: pairs of m values
    for m1 in range(-10, 11, 2):
        for m2 in range(m1 + 2, 11, 2):
            if m1 == 0 or m2 == 0:
                continue
            c_mix = np.zeros(21)
            c_mix[10 + m1] = 1.0 / np.sqrt(2)
            c_mix[10 + m2] = 1.0 / np.sqrt(2)
            starts.append(c_mix)
            c_mix2 = np.zeros(21)
            c_mix2[10 + m1] = 1.0 / np.sqrt(2)
            c_mix2[10 + m2] = -1.0 / np.sqrt(2)
            starts.append(c_mix2)
    
    # Random starts (reduced count, more structured starts above)
    for _ in range(80):
        c = rng.standard_normal(21)
        c = c / np.linalg.norm(c)
        starts.append(c)
        
    constraint = {'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2.0 * c}
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
            
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                start,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            val = _ratio_and_grad(res.x, X, w_flat)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Refine best result with more iterations if time permits
    if best_coeffs is not None and time.time() < _DEADLINE - 60:
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 2000, 'ftol': 1e-13}
            )
            val = _ratio_and_grad(res.x, X, w_flat)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Additional polish pass with even tighter tolerance
    if best_coeffs is not None and time.time() < _DEADLINE - 30:
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 5000, 'ftol': 1e-14}
            )
            val = _ratio_and_grad(res.x, X, w_flat)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final ultra-tight polish
    if best_coeffs is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 10000, 'ftol': 1e-15}
            )
            val = _ratio_and_grad(res.x, X, w_flat)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass

    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (0.01 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e4 / max_abs)
        
    return best_coeffs

# EVOLVE-BLOCK-END
