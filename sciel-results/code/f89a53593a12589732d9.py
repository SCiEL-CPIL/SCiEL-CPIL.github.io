# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize

_DEADLINE = time.time() + 620

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta ** 2))
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2.0 * np.pi / 130
    # Weights for the 2D quadrature
    weights_1d = w_gl * dphi
    return cos_theta, phi, weights_1d

def _compute_harmonics_matrix(cos_theta, phi):
    """
    Compute the matrix X where X[i, j] = Y_j(ct_i, phi_j).
    j ranges from 0 to 20, corresponding to m = -10 to 10.
    """
    N_pts_ct = len(cos_theta)
    N_pts_phi = len(phi)
    X = np.zeros((N_pts_ct * N_pts_phi, 21))
    
    # Meshgrid
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    ct_flat = ct.flatten()
    ph_flat = ph.flatten()
    
    for k, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        # N_m = sqrt(21/(4*pi) * (10-abs_m)! / (10+abs_m)!)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + 
                       gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
        N_m = np.exp(log_N)
        
        # P_10^abs_m(cos theta) with Condon-Shortley phase
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
    R is scale invariant.
    We minimize R.
    Let f = X c.
    N4 = (sum w f^4)^(1/4)
    N2 = (sum w f^2)^(1/2)
    R = N4 / N2
    
    dR/dc = (1/N2) dN4/dc - (N4/N2^2) dN2/dc
    
    dN4/dc = (1/4) N4^-3 * 4 sum w f^3 X = N4^-3 * X^T (w f^3)
    dN2/dc = (1/2) N2^-1 * 2 sum w f X = N2^-1 * X^T (w f)
    
    So dR/dc = (1/N2) N4^-3 X^T (w f^3) - (N4/N2^2) N2^-1 X^T (w f)
             = (1/(N2 N4^3)) X^T (w f^3) - (N4/(N2^3)) X^T (w f)
             
    Alternatively, use log R = 0.5 log N4^2 - log N2.
    log R = 0.25 log(sum w f^4) - 0.5 log(sum w f^2)
    d(log R)/dc = 0.25 * (sum w f^4)^-1 * 4 sum w f^3 X - 0.5 * (sum w f^2)^-1 * 2 sum w f X
                 = (1/S4) X^T (w f^3) - (1/S2) X^T (w f)
    dR/dc = R * d(log R)/dc
    """
    f = X @ c
    w_f = w_flat * f
    w_f2 = w_flat * f * f
    w_f3 = w_flat * f * f * f
    w_f4 = w_flat * f * f * f * f
    
    S2 = np.sum(w_f2)
    S4 = np.sum(w_f4)
    
    if S2 < 1e-10 or S4 < 1e-10:
        return 1e10, np.zeros(21)
        
    N2 = np.sqrt(S2)
    N4 = S4 ** 0.25
    R = N4 / N2
    
    term1 = X.T @ w_f3
    term2 = X.T @ w_f
    
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
    
    # Weights for the flattened vector
    # w_1d is shape (70,). We need to repeat it for each phi point.
    # w_flat[i, j] = w_1d[i] * dphi? No, w_1d already includes dphi?
    # In _build_quadrature: weights_1d = w_gl * dphi.
    # So integral approx is sum_i sum_j w_1d[i] * f(i, j).
    # w_flat should be w_1d repeated 130 times.
    w_flat = np.tile(w_1d, 130)
    
    best_coeffs = None
    best_val = np.inf
    
    # Initial guesses
    # 1. Y_10^0 (m=0)
    # 2. Y_10^10 (m=10)
    # 3. Random
    
    rng = np.random.default_rng(42)
    
    starts = []
    # m=0
    c0 = np.zeros(21)
    c0[10] = 1.0 # m=0 is index 10
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
    
    # m=4, -4
    c4 = np.zeros(21)
    c4[14] = 1.0
    starts.append(c4)
    
    # Random starts
    for _ in range(100):
        c = rng.standard_normal(21)
        c = c / np.linalg.norm(c)
        starts.append(c)
        
    # Optimize from each start
    # We minimize R(c) which is scale invariant.
    # To avoid scaling issues, we can fix ||c||=1 or just let it be.
    # Since R is scale invariant, the gradient is also scale invariant in direction?
    # No, the gradient points in the direction of steepest decrease of R.
    # If we scale c, R doesn't change, but the gradient magnitude scales?
    # Let's just use L-BFGS-B or Nelder-Mead or SLSQP without constraints on norm,
    # but we might drift to 0 or inf.
    # Better to constrain ||c||=1.
    
    constraint = {'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2.0 * c}
    
    for start in starts:
        if time.time() > _DEADLINE - 5:
            break
            
        try:
            # Use SLSQP for constraint handling
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                start,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            val = _ratio_and_grad(res.x, X, w_flat)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception as e:
            pass

    # Refine best result with more iterations if time permits
    if best_coeffs is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(
                lambda c: _ratio_and_grad(c, X, w_flat)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad(c, X, w_flat)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            val = _ratio_and_grad(res.x, X, w_flat)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass

    if best_coeffs is None:
        # Fallback to Y_10^0
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    # Final check and scaling
    # The ratio is scale invariant, so scaling doesn't change R.
    # But we must ensure max|c| is within [1e-3, 1e5].
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (0.01 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e4 / max_abs)
        
    return best_coeffs

# EVOLVE-BLOCK-END
