# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln

_DEADLINE = time.time() + 620

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
    Compute the matrix of real spherical harmonics Y_m at all quadrature points.
    Returns H of shape (N_points, 21).
    m ranges from -10 to 10.
    """
    l = 10
    N_theta = len(cos_theta)
    N_phi = len(phi)
    # H shape: (N_theta, N_phi, 21)
    H = np.zeros((N_theta, N_phi, 21), dtype=np.float64)
    
    # Precompute factorials for N_m
    # N_m = sqrt( (2l+1)/(4*pi) * (l-m)! / (l+m)! )
    # Using log for stability
    log_norm_base = 0.5 * (np.log(2*l + 1) - np.log(4.0 * np.pi))
    
    for m_idx, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        
        # Normalization factor N_m
        log_N = log_norm_base + 0.5 * (gammaln(l - abs_m + 1) - gammaln(l + abs_m + 1))
        N_m = np.exp(log_N)
        
        # Associated Legendre polynomial P_l^m(cos_theta)
        # lpmv includes Condon-Shortley phase (-1)^m
        P = lpmv(abs_m, l, cos_theta) # shape (N_theta,)
        
        if m == 0:
            Y = N_m * P
            H[:, :, m_idx] = Y[:, np.newaxis] * np.ones(N_phi)[np.newaxis, :]
        elif m > 0:
            # Y_m = sqrt(2) N_m P cos(m phi)
            Y = np.sqrt(2.0) * N_m * P
            cos_m_phi = np.cos(m * phi) # shape (N_phi,)
            H[:, :, m_idx] = Y[:, np.newaxis] * cos_m_phi[np.newaxis, :]
        else:
            # Y_m = sqrt(2) N_|m| P sin(|m| phi)
            Y = np.sqrt(2.0) * N_m * P
            sin_abs_m_phi = np.sin(abs_m * phi) # shape (N_phi,)
            H[:, :, m_idx] = Y[:, np.newaxis] * sin_abs_m_phi[np.newaxis, :]
            
    # Flatten to (N_points, 21)
    H_flat = H.reshape(-1, 21)
    return H_flat

def _compute_ratio(coeffs, H_flat, weights_flat):
    """
    Compute ||f||_4 / ||f||_2.
    coeffs: shape (21,)
    H_flat: shape (N_points, 21)
    weights_flat: shape (N_points,)
    """
    f = H_flat @ coeffs
    f2 = f ** 2
    f4 = f2 ** 2
    
    norm2_sq = np.sum(weights_flat * f2)
    norm4_4 = np.sum(weights_flat * f4)
    
    if norm2_sq <= 1e-30:
        return np.inf, np.inf, np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2, norm2_sq, norm4_4

def _objective_and_grad(coeffs, H_flat, weights_flat):
    """
    Minimize ||f||_4^4 subject to ||c||_2 = 1.
    This is equivalent to minimizing ||f||_4 / ||f||_2 when ||f||_2 is fixed to 1.
    """
    f = H_flat @ coeffs
    f2 = f ** 2
    f3 = f2 * f
    f4 = f2 ** 2
    
    # Objective: sum(w * f^4)
    obj = np.sum(weights_flat * f4)
    
    # Gradient of obj w.r.t coeffs
    # d(obj)/d(c_m) = sum_j w_j * 4 * f_j^3 * H_{j,m}
    # grad = 4 * H_flat.T @ (weights_flat * f3)
    grad = 4.0 * (H_flat.T @ (weights_flat * f3))
    
    return obj, grad

def _power_iteration_eigen(coeffs, H_flat, weights_flat, max_iter=500, tol=1e-10):
    """
    Use power iteration on the linear operator T(c) = grad(||f||_4^4) to find the
    direction that maximizes the 4th moment. This is a structural change from
    gradient descent to an eigenvalue-like iteration.
    """
    c = coeffs.copy()
    c /= np.linalg.norm(c)
    
    prev_obj = np.inf
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        obj, grad = _objective_and_grad(c, H_flat, weights_flat)
        
        # In power iteration for maximizing a quadratic form, we move in the
        # direction of the gradient (eigenvector direction) and normalize.
        # Here, we are maximizing the 4th moment, so we move towards the gradient.
        c_new = grad
        
        norm_c = np.linalg.norm(c_new)
        if norm_c < 1e-12:
            c_new = c
        else:
            c_new /= norm_c
            
        # Check convergence
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj
        c = c_new
        
    return c

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    # 1. Setup
    cos_theta, phi, weights_1d = _build_quadrature()
    N_theta = len(cos_theta)
    N_phi = len(phi)
    
    # Weights for flat array
    weights_flat = np.repeat(weights_1d, N_phi)
    
    # 2. Precompute Harmonics
    H_flat = _compute_harmonics_matrix(cos_theta, phi)
    
    # 3. Optimization
    # We minimize J(c) = sum w_j (H c)_j^4 subject to ||c||_2 = 1.
    # This minimizes the 4th moment. Since ||f||_2 = 1, R = ||f||_4.
    
    best_obj = np.inf
    best_coeffs = None
    
    # Multiple random starts
    rng = np.random.default_rng(42)
    
    num_starts = 100
    max_iter_pd = 2000
    lr = 0.005 # Learning rate, might need tuning
    
    for i in range(num_starts):
        if time.time() > _DEADLINE - 10:
            break
            
        # Random start on the sphere
        c = rng.standard_normal(21)
        c /= np.linalg.norm(c)
        
        # Projected Gradient Descent
        prev_obj = np.inf
        for it in range(max_iter_pd):
            if time.time() > _DEADLINE - 5:
                break
                
            obj, grad = _objective_and_grad(c, H_flat, weights_flat)
            
            # Update
            c_new = c - lr * grad
            
            # Normalize
            norm_c = np.linalg.norm(c_new)
            if norm_c < 1e-12:
                c_new = c
            else:
                c_new /= norm_c
                
            # Check convergence
            if abs(prev_obj - obj) < 1e-12:
                break
            prev_obj = obj
            c = c_new
            
        # Final evaluation
        obj, grad = _objective_and_grad(c, H_flat, weights_flat)
        if obj < best_obj:
            best_obj = obj
            best_coeffs = c.copy()
            
    # If no good result, fallback
    if best_coeffs is None:
        best_coeffs = np.ones(21) / np.sqrt(21.0)
        
    # Ensure validity
    result = best_coeffs.astype(np.float64)
    if result.shape != (21,):
        result = np.ones(21, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        result = np.ones(21, dtype=np.float64)
        
    max_abs = np.max(np.abs(result))
    if max_abs < 1e-3:
        result = result / max_abs * 0.01 if max_abs > 0 else np.ones(21, dtype=np.float64)
    elif max_abs > 1e5:
        result = result / max_abs * 1e4
        
    return result

# EVOLVE-BLOCK-END
