# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import lpmv, factorial
import time


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
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * 
                      float(factorial(k - m_abs) / factorial(k + m_abs)))
        P_m = lpmv(m_abs, k, cos_theta)
        
        if m_abs == 0:
            Y[0] = N_m * P_m[:, np.newaxis] * np.ones((1, n_phi))
        else:
            Y[k + m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.cos(m_abs * phi)[np.newaxis, :]
            Y[k - m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.sin(m_abs * phi)[np.newaxis, :]
    
    return Y


def _compute_ratio(coeffs, Y, weights):
    """Compute R = ||f||_4 / ||f||_2 using precomputed harmonics and weights."""
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    return norm4 / norm2


def _compute_ratio_and_grad(coeffs, Y, weights):
    """Compute R and its gradient with respect to coeffs (vectorized)."""
    n_theta = Y.shape[1]
    n_phi = Y.shape[2]
    n_modes = Y.shape[0]
    
    # Flatten Y for efficient matrix operations
    Y_flat = Y.reshape(n_modes, -1)  # shape (n_modes, n_points)
    w_flat = weights.ravel()
    
    f = Y_flat @ coeffs  # shape (n_points,)
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(w_flat * f2)
    norm4_sq = np.sum(w_flat * f4)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros(n_modes)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    R = norm4 / norm2
    
    if norm4_sq <= 0:
        return R, np.zeros(n_modes)
    
    # Vectorized gradient computation
    # d_norm4_sq_dc_m = 4 * sum(w * f^3 * Y_m) for all m at once
    f3 = f * f2
    d_norm4_sq_dc = 4.0 * (Y_flat * (w_flat * f3)[np.newaxis, :]).sum(axis=1)
    
    # d_norm2_sq_dc_m = 2 * sum(w * f * Y_m) for all m at once
    d_norm2_sq_dc = 2.0 * (Y_flat * (w_flat * f)[np.newaxis, :]).sum(axis=1)
    
    d_norm4_dc = (1.0 / (4.0 * norm4**3)) * d_norm4_sq_dc
    d_norm2_dc = (1.0 / (2.0 * norm2)) * d_norm2_sq_dc
    
    grad = d_norm4_dc / norm2 - norm4 * d_norm2_dc / (norm2 * norm2)
    
    return R, grad


def _random_unit_vector(rng, n=21):
    """Generate a random unit vector."""
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    return v


def _try_symmetric_patterns(Y, weights, deadline, safety_margin):
    """Try various symmetric coefficient patterns."""
    best_coeffs = None
    best_ratio = np.inf
    
    patterns = []
    
    # All ones normalized
    c = np.ones(21) / np.sqrt(21)
    patterns.append(c)
    
    # Only m=0
    c = np.zeros(21)
    c[10] = 1.0
    patterns.append(c)
    
    # m=0 and m=±10
    c = np.zeros(21)
    c[10] = 1.0  # m=0
    c[0] = 1.0   # m=-10
    c[20] = 1.0  # m=+10
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Even m modes: 0, 2, 4, 6, 8, 10
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10 + m] = 1.0
        if m > 0:
            c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Odd m modes: 1, 3, 5, 7, 9
    c = np.zeros(21)
    for m in [1, 3, 5, 7, 9]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Cosine-weighted: c_m = cos(m * pi/20) for m=0..10
    c = np.zeros(21)
    for m in range(11):
        c[10 + m] = np.cos(m * np.pi / 20.0)
        if m > 0:
            c[10 - m] = np.cos(m * np.pi / 20.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Sine-weighted
    c = np.zeros(21)
    for m in range(1, 11):
        c[10 + m] = np.sin(m * np.pi / 21.0)
        c[10 - m] = np.sin(m * np.pi / 21.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Try to reduce peaks: emphasize low |m| modes
    c = np.zeros(21)
    for m in range(11):
        w = 1.0 / (1.0 + 0.1 * m)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Quadratic falloff
    c = np.zeros(21)
    for m in range(11):
        w = 1.0 / (1.0 + 0.05 * m**2)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    for c in patterns:
        r = _compute_ratio(c, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c.copy()
    
    return best_coeffs, best_ratio


def _polish_coefficients(c, Y, weights, deadline, safety_margin, max_rounds=200):
    """High-precision polish: alternating coordinate descent with line search."""
    c = c.copy()
    c /= np.linalg.norm(c)
    best_r = _compute_ratio(c, Y, weights)
    
    for rnd in range(max_rounds):
        if time.time() > deadline - safety_margin:
            break
        improved = False
        for m in range(21):
            if time.time() > deadline - safety_margin:
                break
            # Line search along coordinate m
            c_old = c.copy()
            best_local_r = best_r
            best_local_c = c_old.copy()
            
            for delta in np.concatenate([
                np.linspace(-0.5, 0.5, 21)[1:-1],
                np.linspace(-0.1, 0.1, 11)[1:-1]
            ]):
                c_new = c_old.copy()
                c_new[m] += delta
                nrm = np.linalg.norm(c_new)
                if nrm < 1e-12:
                    continue
                c_new /= nrm
                r_new = _compute_ratio(c_new, Y, weights)
                if r_new < best_local_r - 1e-12:
                    best_local_r = r_new
                    best_local_c = c_new.copy()
            
            if best_local_r < best_r - 1e-12:
                c = best_local_c
                best_r = best_local_r
                improved = True
        
        if not improved:
            break
    
    return c, best_r


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 60.0  # seconds
    
    # Build quadrature grid
    cos_theta, phi, weights = _build_quadrature()
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, k=10)
    
    # Try symmetric patterns first
    best_coeffs, best_ratio = _try_symmetric_patterns(Y, weights, deadline, safety_margin)
    
    if best_coeffs is None:
        best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
        best_ratio = _compute_ratio(best_coeffs, Y, weights)
    
    # Set up optimization
    rng = np.random.default_rng(42)
    
    def objective_and_grad(c):
        if time.time() > deadline - safety_margin:
            return _compute_ratio(best_coeffs, Y, weights), np.zeros(21)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros(21)
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
        # Project gradient to sphere constraint
        grad_proj = grad - np.dot(grad, c_normalized) * c_normalized
        return R, grad_proj
    
    # Multiple restarts with different strategies
    n_restarts = 40
    
    for i in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break
        
        # Choose initial guess
        if i == 0:
            c_init = best_coeffs.copy()
        elif i == 1:
            c_init = np.zeros(21)
            c_init[10] = 1.0  # m=0 only
        elif i == 2:
            c_init = np.zeros(21)
            c_init[20] = 1.0  # m=10 cos
        elif i == 3:
            c_init = np.zeros(21)
            c_init[0] = 1.0   # m=-10 sin
        elif i == 4:
            c_init = np.zeros(21)
            c_init[10:16] = 1.0  # m=0..5 cos
            c_init /= np.linalg.norm(c_init)
        elif i == 5:
            # Emphasize low m
            c_init = np.zeros(21)
            for m in range(11):
                w = 1.0 / (1.0 + 0.2 * m)
                c_init[10 + m] = w
                if m > 0:
                    c_init[10 - m] = w
            c_init /= np.linalg.norm(c_init)
        elif i == 6:
            # Emphasize high m
            c_init = np.zeros(21)
            for m in range(11):
                w = 1.0 / (1.0 + 0.2 * (10 - m))
                c_init[10 + m] = w
                if m > 0:
                    c_init[10 - m] = w
            c_init /= np.linalg.norm(c_init)
        else:
            c_init = _random_unit_vector(rng)
        
        # Run L-BFGS-B with gradient
        def obj(c):
            r, _ = objective_and_grad(c)
            return r
        
        def grad(c):
            _, g = objective_and_grad(c)
            return g
        
        result = minimize(
            obj,
            c_init,
            jac=grad,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt
    
    # Additional refinement: coordinate descent style
    if time.time() < deadline - safety_margin:
        c = best_coeffs.copy()
        for _ in range(80):
            if time.time() > deadline - safety_margin:
                break
            improved = False
            for m in range(21):
                if time.time() > deadline - safety_margin:
                    break
                # Try perturbing component m with finer steps
                for delta in np.concatenate([
                    np.linspace(-0.3, 0.3, 13)[1:-1],
                    np.linspace(-0.05, 0.05, 11)[1:-1]
                ]):
                    c_new = c.copy()
                    c_new[m] += delta
                    c_new /= np.linalg.norm(c_new)
                    r_new = _compute_ratio(c_new, Y, weights)
                    if r_new < best_ratio - 1e-12:
                        c = c_new
                        best_ratio = r_new
                        improved = True
                        break
            if not improved:
                break
    
    # Final high-precision polish
    if time.time() < deadline - safety_margin:
        best_coeffs, best_ratio = _polish_coefficients(
            best_coeffs, Y, weights, deadline, safety_margin, max_rounds=100
        )
    
    # Ensure output is valid
    best_coeffs = best_coeffs.astype(np.float64)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        target = 1.0
        best_coeffs = best_coeffs * (target / max_abs)
    
    return best_coeffs


# EVOLVE-BLOCK-END
