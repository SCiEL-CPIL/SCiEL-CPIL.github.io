# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize


def _build_quadrature():
    """Build the 70-point Gauss-Legendre (cos theta) x 130-point phi grid."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    weights = np.outer(w_gl, np.full(n_phi, dphi))
    cos_theta = x_gl
    return cos_theta, phi, weights


def _spherical_harmonics_real(cos_theta, phi, k=10):
    """
    Compute real orthonormal spherical harmonics of degree k.
    Returns array of shape (2*k+1, n_theta, n_phi) where index i corresponds to m = i - k.
    """
    n_theta = len(cos_theta)
    n_phi = len(phi)
    Y = np.zeros((2 * k + 1, n_theta, n_phi))
    
    for m_abs in range(k + 1):
        log_N_sq = np.log((2 * k + 1) / (4 * np.pi)) + gammaln(k + 1 - m_abs) - gammaln(k + 1 + m_abs)
        N_m = np.sqrt(np.exp(log_N_sq))
        
        P_m = lpmv(m_abs, k, cos_theta)
        
        if m_abs == 0:
            Y[k] = N_m * P_m[:, np.newaxis] * np.ones((1, n_phi))
        else:
            Y[k + m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.cos(m_abs * phi)[np.newaxis, :]
            Y[k - m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.sin(m_abs * phi)[np.newaxis, :]
    
    return Y


def _compute_ratio_and_grad(coeffs, Y, weights):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient with respect to coeffs.
    """
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    
    if norm2_sq <= 1e-15:
        return np.inf, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    R = norm4 / norm2
    
    weighted_f3 = weights * (f * f * f)
    weighted_f = weights * f
    
    integral_f3_Y = np.tensordot(weighted_f3, Y, axes=([0, 1], [1, 2]))
    integral_f_Y = np.tensordot(weighted_f, Y, axes=([0, 1], [1, 2]))
    
    term1 = integral_f3_Y / (norm4**3 * norm2)
    term2 = (norm4 / norm2**3) * integral_f_Y
    
    grad = term1 - term2
    
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 15.0
    
    # Build quadrature grid
    cos_theta, phi, weights = _build_quadrature()
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, k=10)
    
    best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
    best_ratio, _ = _compute_ratio_and_grad(best_coeffs, Y, weights)
    
    rng = np.random.default_rng(42)
    initial_guesses = []
    
    # 1. All ones
    c0 = np.ones(21) / np.sqrt(21)
    initial_guesses.append(c0)
    
    # 2. Pure m=0
    c1 = np.zeros(21)
    c1[10] = 1.0
    initial_guesses.append(c1)
    
    # 3. Pure m=10 cos
    c2 = np.zeros(21)
    c2[20] = 1.0
    initial_guesses.append(c2)
    
    # 4. Pure m=10 sin
    c3 = np.zeros(21)
    c3[0] = 1.0
    initial_guesses.append(c3)
    
    # 5. Mix of m=0 and m=10
    c4 = np.zeros(21)
    c4[10] = 1.0
    c4[20] = 1.0
    c4 /= np.linalg.norm(c4)
    initial_guesses.append(c4)
    
    # 6. Mix of even m
    for m_val in [0, 2, 4, 6, 8, 10]:
        c_temp = np.zeros(21)
        if m_val == 0:
            c_temp[10] = 1.0
        else:
            c_temp[10 + m_val] = 1.0
            c_temp[10 - m_val] = 1.0
        c_temp /= np.linalg.norm(c_temp)
        initial_guesses.append(c_temp)
    
    # 7. Additional structured starts: concentrate on low |m|
    for m_set in [[0], [0, 2], [0, 2, 4], [2, 4], [2, 4, 6], [4, 6, 8]]:
        c_temp = np.zeros(21)
        for m in m_set:
            if m == 0:
                c_temp[10] = 1.0
            else:
                c_temp[10 + m] = 1.0
                c_temp[10 - m] = 1.0
        if np.linalg.norm(c_temp) > 1e-10:
            c_temp /= np.linalg.norm(c_temp)
            initial_guesses.append(c_temp)
    
    # 8. Phase-shifted versions of m=10 (cos and sin with offsets)
    for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        c_temp = np.zeros(21)
        c_temp[20] = np.cos(phase)
        c_temp[0] = np.sin(phase)
        c_temp /= np.linalg.norm(c_temp)
        initial_guesses.append(c_temp)
    
    # 9. More random starts
    for i in range(80):
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        initial_guesses.append(c_rand)
    
    # Evaluate initial guesses
    for c_init in initial_guesses:
        r, _ = _compute_ratio_and_grad(c_init, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c_init.copy()
    
    # Optimization function
    def objective_and_grad(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros_like(c)
        
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
        
        # Project gradient onto the tangent space of the sphere
        dot = np.dot(grad, c_normalized)
        grad_c = (grad - c_normalized * dot) / c_norm
        
        return R, grad_c

    # Run optimization from best initial guess
    result = minimize(
        objective_and_grad,
        best_coeffs,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
    )
    
    if result.fun < best_ratio:
        c_opt = result.x / np.linalg.norm(result.x)
        r_opt, _ = _compute_ratio_and_grad(c_opt, Y, weights)
        if r_opt < best_ratio:
            best_ratio = r_opt
            best_coeffs = c_opt.copy()
    
    # Try more random restarts
    n_restarts = 60
    for i in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break
            
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        
        max_iter = 1500
        if i < 15:
            max_iter = 3000
            ftol = 1e-16
            gtol = 1e-14
        else:
            ftol = 1e-15
            gtol = 1e-12
            
        result = minimize(
            objective_and_grad,
            c_rand,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 80}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt, _ = _compute_ratio_and_grad(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
                
    # Final high-precision polish
    if time.time() < deadline - safety_margin:
        result_final = minimize(
            objective_and_grad,
            best_coeffs,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 200}
        )
        if result_final.fun < best_ratio:
            c_opt = result_final.x / np.linalg.norm(result.x)
            r_opt, _ = _compute_ratio_and_grad(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
    
    # Ensure output is valid
    best_coeffs = best_coeffs.astype(np.float64)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
        
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        target = 1.0
        best_coeffs = best_coeffs * (target / max_abs)
        
    return best_coeffs


# EVOLVE-BLOCK-END
