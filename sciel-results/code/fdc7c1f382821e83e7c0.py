# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize, differential_evolution


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


def _build_fourth_power_kernel(Y, weights):
    """
    Build the matrix K such that ||f||_4^4 = sum_{i,j} c_i^2 K_{ij} c_j^2
    where f = sum c_i Y_i and K_{ij} = sum_p w_p Y_i(p)^2 Y_j(p)^2.
    """
    n_modes = Y.shape[0]
    n_pts = Y.shape[1] * Y.shape[2]
    
    # Flatten Y to (n_modes, n_pts)
    Y_flat = Y.reshape(n_modes, n_pts)
    w_flat = weights.reshape(n_pts)
    
    # K[i,j] = sum_p w_p * Y_i(p)^2 * Y_j(p)^2
    # = (Y_flat^2 * w_flat) @ (Y_flat^2)^T
    Y_sq_w = Y_flat * Y_flat * w_flat[np.newaxis, :]
    K = Y_sq_w @ Y_flat.T
    
    return K


def _compute_ratio_and_grad(coeffs, Y, weights, K):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient using the spectral kernel.
    coeffs: shape (21,)
    Y: shape (21, n_theta, n_phi)
    weights: shape (n_theta, n_phi)
    K: shape (21, 21) - fourth power kernel
    """
    n_modes = len(coeffs)
    c = coeffs
    
    # ||f||_2^2 = c^T G c where G is the Gram matrix (should be identity for orthonormal)
    # But let's compute directly for safety
    f = np.tensordot(c, Y, axes=([0], [0]))
    f2 = f * f
    norm2_sq = np.sum(weights * f2)
    
    # ||f||_4^4 = sum_{i,j} c_i^2 K_{ij} c_j^2
    c_sq = c * c
    norm4_4 = c_sq @ K @ c_sq
    
    if norm2_sq <= 1e-15 or norm4_4 <= 1e-15:
        return np.inf, np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    R = norm4 / norm2
    
    # Gradient:
    # dR/dc_k = (1/(norm4^3 * norm2)) * sum_j c_j^2 * 2 * K[k,j] * c_k  -  (norm4/norm2^3) * (G c)_k
    # Wait, let me re-derive.
    # R = norm4 / norm2
    # dR/dc_k = (1/norm2) * dnorm4/dc_k - (norm4/norm2^2) * dnorm2/dc_k
    # 
    # norm4 = (norm4_4)^(1/4)
    # dnorm4/dc_k = (1/4) * norm4_4^(-3/4) * dnorm4_4/dc_k
    # dnorm4_4/dc_k = 2 * c_k * (K @ c_sq)_k
    # 
    # norm2 = (norm2_sq)^(1/2)
    # dnorm2/dc_k = (1/2) * norm2_sq^(-1/2) * dnorm2_sq/dc_k
    # dnorm2_sq/dc_k = 2 * (G @ c)_k  where G is the Gram matrix
    # For orthonormal basis, G = I, so dnorm2_sq/dc_k = 2*c_k
    # But let's be general.
    
    # Actually for orthonormal harmonics with the quadrature, the Gram matrix should be ~I
    # Let me just compute it directly to be safe
    
    # dnorm4_4/dc_k = 2 * c_k * (K @ c_sq)_k
    Kc_sq = K @ c_sq
    dnorm4_4_dc = 2.0 * c * Kc_sq
    
    dnorm4_dc = 0.25 * norm4_4 ** (-0.75) * dnorm4_4_dc
    
    # dnorm2_sq/dc_k: compute using the quadrature
    # norm2_sq = sum_p w_p * f(p)^2
    # dnorm2_sq/dc_k = 2 * sum_p w_p * f(p) * Y_k(p)
    weighted_f = weights * f
    dnorm2_sq_dc = 2.0 * np.tensordot(weighted_f, Y, axes=([0, 1], [1, 2]))
    
    dnorm2_dc = 0.5 / norm2 * dnorm2_sq_dc
    
    dR_dc = dnorm4_dc / norm2 - norm4 / norm2_sq * dnorm2_dc
    
    return R, dR_dc


def _compute_ratio_only(coeffs, Y, weights, K):
    """Compute only the ratio R, without gradient, using the kernel."""
    c = coeffs
    f = np.tensordot(c, Y, axes=([0], [0]))
    f2 = f * f
    norm2_sq = np.sum(weights * f2)
    
    c_sq = c * c
    norm4_4 = c_sq @ K @ c_sq
    
    if norm2_sq <= 1e-15 or norm4_4 <= 1e-15:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 45.0
    
    # Build quadrature grid
    cos_theta, phi, weights = _build_quadrature()
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, k=10)
    
    # Precompute the fourth-power kernel for fast L4 norm evaluation
    K = _build_fourth_power_kernel(Y, weights)
    
    best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
    best_ratio, _ = _compute_ratio_and_grad(best_coeffs, Y, weights, K)
    
    # Generate initial guesses
    rng = np.random.default_rng(42)
    initial_guesses = []
    
    c0 = np.ones(21) / np.sqrt(21)
    initial_guesses.append(c0)
    
    c1 = np.zeros(21)
    c1[10] = 1.0
    initial_guesses.append(c1)
    
    c2 = np.zeros(21)
    c2[20] = 1.0
    initial_guesses.append(c2)
    
    c3 = np.zeros(21)
    c3[0] = 1.0
    initial_guesses.append(c3)
    
    c4 = np.zeros(21)
    c4[10] = 1.0
    c4[20] = 1.0
    c4 /= np.linalg.norm(c4)
    initial_guesses.append(c4)
    
    for m_val in [0, 2, 4, 6, 8, 10]:
        c_temp = np.zeros(21)
        if m_val == 0:
            c_temp[10] = 1.0
        else:
            c_temp[10 + m_val] = 1.0
            c_temp[10 - m_val] = 1.0
        c_temp /= np.linalg.norm(c_temp)
        initial_guesses.append(c_temp)
        
    for i in range(50):
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        initial_guesses.append(c_rand)
        
    for c_init in initial_guesses:
        r = _compute_ratio_only(c_init, Y, weights, K)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c_init.copy()
    
    # Optimization function with gradient
    def objective_and_grad(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros_like(c)
        
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights, K)
        
        # Project gradient onto the tangent space of the sphere
        dot = np.dot(grad, c_normalized)
        grad_c = (grad - c_normalized * dot) / c_norm
        
        return R, grad_c

    # Phase 1: Differential Evolution for global search
    def de_objective(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10
        c_normalized = c / c_norm
        return _compute_ratio_only(c_normalized, Y, weights, K)
    
    bounds = [(-1.0, 1.0)] * 21
    
    de_result = differential_evolution(
        de_objective,
        bounds,
        seed=42,
        maxiter=50,
        popsize=15,
        tol=1e-10,
        polish=False,
        init='sobol'
    )
    
    if de_result.fun < best_ratio:
        c_de = de_result.x / np.linalg.norm(de_result.x)
        r_de = _compute_ratio_only(c_de, Y, weights, K)
        if r_de < best_ratio:
            best_ratio = r_de
            best_coeffs = c_de.copy()
    
    # Phase 2: L-BFGS-B polishing from best DE result
    if time.time() < deadline - safety_margin:
        result = minimize(
            objective_and_grad,
            best_coeffs,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio_only(c_opt, Y, weights, K)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
    
    # Phase 3: More random restarts with L-BFGS-B
    for i in range(80):
        if time.time() > deadline - safety_margin:
            break
            
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        
        max_iter = 1500
        if i < 20:
            max_iter = 2500
            
        result = minimize(
            objective_and_grad,
            c_rand,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio_only(c_opt, Y, weights, K)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
                
    # Phase 4: High-precision polish pass
    if time.time() < deadline - safety_margin:
        result_final = minimize(
            objective_and_grad,
            best_coeffs,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 8000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
        )
        if result_final.fun < best_ratio:
            c_opt = result_final.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio_only(c_opt, Y, weights, K)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
    
    # Phase 5: Additional DE runs from different seeds if time allows
    if time.time() < deadline - safety_margin - 60:
        for seed in [123, 456, 789]:
            if time.time() > deadline - safety_margin:
                break
            try:
                de_result2 = differential_evolution(
                    de_objective,
                    bounds,
                    seed=seed,
                    maxiter=30,
                    popsize=12,
                    tol=1e-10,
                    polish=False,
                    init='sobol'
                )
                if de_result2.fun < best_ratio:
                    c_de2 = de_result2.x / np.linalg.norm(de_result2.x)
                    r_de2 = _compute_ratio_only(c_de2, Y, weights, K)
                    if r_de2 < best_ratio:
                        best_ratio = r_de2
                        best_coeffs = c_de2.copy()
                        
                        # Polish the new best
                        if time.time() < deadline - safety_margin:
                            result_polish = minimize(
                                objective_and_grad,
                                best_coeffs,
                                jac=True,
                                method='L-BFGS-B',
                                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
                            )
                            if result_polish.fun < best_ratio:
                                c_opt = result_polish.x / np.linalg.norm(result_polish.x)
                                r_opt = _compute_ratio_only(c_opt, Y, weights, K)
                                if r_opt < best_ratio:
                                    best_ratio = r_opt
                                    best_coeffs = c_opt.copy()
            except Exception:
                pass
    
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
