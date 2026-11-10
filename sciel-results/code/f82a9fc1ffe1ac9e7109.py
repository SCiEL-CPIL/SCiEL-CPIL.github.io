# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_theta = w
    w_phi = 2 * np.pi / n_phi
    return x, w_theta, phi, w_phi

def _spherical_harmonics(x, phi, degree=10):
    """
    Compute real orthonormal spherical harmonics of given degree.
    Returns array of shape (n_theta, n_phi, 2*degree+1) where index m corresponds to m-degree.
    """
    k = degree
    n_theta = len(x)
    n_phi = len(phi)
    
    # Precompute normalization constants
    norm_constants = np.zeros(k + 1)
    for m in range(k + 1):
        ratio = 1.0
        for j in range(10 - m + 1, 10 + m + 1):
            ratio /= j
        norm_constants[m] = np.sqrt(21.0 / (4.0 * np.pi) * ratio)
    
    # Associated Legendre polynomials
    P = np.zeros((n_theta, k + 1))
    for m in range(k + 1):
        P[:, m] = lpmv(m, k, x)
    
    # Build Y array
    Y = np.zeros((n_theta, n_phi, 2 * k + 1))
    
    # m = 0
    Y[:, :, k] = norm_constants[0] * P[:, 0][:, np.newaxis]
    
    # m > 0 (cos) and m < 0 (sin)
    for m in range(1, k + 1):
        cos_term = np.cos(m * phi)[np.newaxis, :]
        sin_term = np.sin(m * phi)[np.newaxis, :]
        P_m = P[:, m][:, np.newaxis]
        coeff = np.sqrt(2.0) * norm_constants[m]
        
        Y[:, :, k + m] = coeff * P_m * cos_term
        Y[:, :, k - m] = coeff * P_m * sin_term
    
    return Y

def _compute_ratio_and_grad(coeffs, Y, w_theta, w_phi):
    """
    Compute L4/L2 ratio and its gradient.
    """
    f = np.tensordot(coeffs, Y, axes=([0], [2]))
    
    w_full = w_theta[:, np.newaxis] * w_phi
    
    f2 = f * f
    norm2_sq = np.sum(w_full * f2)
    
    f4 = f2 * f2
    norm4 = np.sum(w_full * f4)
    
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4_val = norm4 ** 0.25
    
    ratio = norm4_val / norm2
    
    f3 = f2 * f
    sum_w_f3Y = np.einsum('ij,ijh->h', w_full * f3, Y)
    sum_w_fY = np.einsum('ij,ijh->h', w_full * f, Y)
    
    A = norm4_val
    B = norm2
    
    grad = A**(-3) * B**(-1) * sum_w_f3Y - A * B**(-3) * sum_w_fY
    
    return ratio, grad

def _fixed_point_polish(c, Y, w_theta, w_phi, max_iter=200, tol=1e-14):
    """
    Fixed point iteration: c <- normalize(int f^3 Y).
    This finds eigenvectors of the quartic form.
    """
    c = c.copy()
    c = c / np.linalg.norm(c)
    
    for _ in range(max_iter):
        f = np.tensordot(c, Y, axes=([0], [2]))
        w_full = w_theta[:, np.newaxis] * w_phi
        S = np.einsum('ij,ijh->h', w_full * (f * f * f), Y)
        
        S_norm = np.linalg.norm(S)
        if S_norm < 1e-15:
            break
            
        c_new = S / S_norm
        
        diff = np.linalg.norm(c_new - c)
        if diff < tol:
            c = c_new
            break
            
        c = c_new
        
    return c

def _projected_newton_step(c, Y, w_theta, w_phi, max_iter=100, tol=1e-14):
    """
    Projected Newton method on the unit sphere.
    Minimize R(c) subject to ||c|| = 1.
    """
    c = c.copy()
    c = c / np.linalg.norm(c)
    
    f_val, grad = _compute_ratio_and_grad(c, Y, w_theta, w_phi)
    
    # Project gradient onto tangent space
    grad_tan = grad - np.dot(grad, c) * c
    
    for it in range(max_iter):
        if np.linalg.norm(grad_tan) < tol:
            break
        
        f = np.tensordot(c, Y, axes=([0], [2]))
        w_full = w_theta[:, np.newaxis] * w_phi
        f2 = f * f
        
        norm2_sq = np.sum(w_full * f2)
        norm4_sq = np.sum(w_full * f2 * f2)
        
        if norm2_sq < 1e-30 or norm4_sq < 1e-30:
            break
            
        norm2 = np.sqrt(norm2_sq)
        norm4 = norm4_sq ** 0.25
        
        f3 = f2 * f
        s_f3Y = np.einsum('ij,ijh->h', w_full * f3, Y)
        s_fY = np.einsum('ij,ijh->h', w_full * f, Y)
        
        A = norm4
        B = norm2
        
        H_f2YY = np.einsum('ij,ijk,ijl->kl', w_full * f2, Y, Y)
        H_YY = np.einsum('ij,ijk,ijl->kl', w_full, Y, Y)
        
        dA = A**(-3) * s_f3Y
        dB = B**(-1) * s_fY
        
        d2A = -3 * A**(-4) * np.outer(dA, s_f3Y) + 3 * A**(-3) * H_f2YY
        d2B = -B**(-2) * np.outer(dB, s_fY) + B**(-1) * H_YY
        
        H = (1.0/B) * d2A - (A/(B**2)) * d2B + (2.0*A - 2.0*B**2)/(B**3) * np.outer(dA, dB)
        
        P = np.eye(21) - np.outer(c, c)
        H_tan = P @ H @ P
        
        # Solve with damping
        try:
            d = np.linalg.solve(H_tan + 1e-10 * np.eye(21), -grad_tan)
        except np.linalg.LinAlgError:
            d = -grad_tan
            
        d = d - np.dot(d, c) * c
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            break
        d = d / d_norm
        
        # Line search
        best_c = c
        best_f = f_val
        best_alpha = 0.1
        c1 = 1e-4
        
        alpha = 1.0
        for _ in range(20):
            c_new = c + alpha * d
            norm_new = np.linalg.norm(c_new)
            if norm_new < 1e-10:
                alpha *= 0.5
                continue
            c_new = c_new / norm_new
            
            f_new, _ = _compute_ratio_and_grad(c_new, Y, w_theta, w_phi)
            
            if f_new <= f_val - c1 * alpha * np.dot(grad_tan, d):
                best_f = f_new
                best_c = c_new
                best_alpha = alpha
                break
            alpha *= 0.5
        
        if best_f >= f_val:
            break
        
        c = best_c
        f_val = best_f
        f_val, grad = _compute_ratio_and_grad(c, Y, w_theta, w_phi)
        grad_tan = grad - np.dot(grad, c) * c
    
    return c, f_val

def _generate_starts():
    """Generate diverse starting points."""
    starts = []
    
    # All ones
    s = np.ones(21, dtype=np.float64)
    s = s / np.linalg.norm(s)
    starts.append(s)
    
    # Emphasize m=0
    s = np.zeros(21)
    s[10] = 1.0
    s = s / np.linalg.norm(s)
    starts.append(s)
    
    # Emphasize each m value
    for m in range(1, 11):
        s = np.zeros(21)
        s[10 + m] = 1.0
        s[10 - m] = 1.0
        s = s / np.linalg.norm(s)
        starts.append(s)
        
    # Pairs of m values
    for m1 in range(1, 11):
        for m2 in range(m1 + 1, 11):
            s = np.zeros(21)
            s[10 + m1] = 1.0
            s[10 - m1] = 1.0
            s[10 + m2] = 0.7
            s[10 - m2] = 0.7
            s = s / np.linalg.norm(s)
            starts.append(s)
            
    # Random starts
    np.random.seed(42)
    for _ in range(20):
        s = np.random.randn(21)
        s = s / np.linalg.norm(s)
        starts.append(s)
        
    return starts

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Returns np.ndarray of shape (21,) with coefficients c_m for m = -10..10.
    """
    deadline = time.time() + 600
    
    # Build quadrature
    x, w_theta, phi, w_phi = _build_quadrature()
    Y = _spherical_harmonics(x, phi, degree=10)
    
    # Initial guess: all ones, normalized
    x0 = np.ones(21, dtype=np.float64)
    x0 = x0 / np.linalg.norm(x0)
    
    best_coeffs = x0.copy()
    best_ratio = _compute_ratio_and_grad(x0, Y, w_theta, w_phi)[0]
    
    # Phase 1: Multiple starting points
    starts = _generate_starts()
    
    for s in starts:
        if time.time() > deadline - 120:
            break
            
        # L-BFGS-B with tight tolerance
        try:
            result = minimize(
                lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0],
                s,
                jac=lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x
        except Exception:
            pass
            
        # Follow up with L4 Newton
        if time.time() < deadline - 90:
            c_new, _ = _projected_newton_step(best_coeffs, Y, w_theta, w_phi, max_iter=50, tol=1e-13)
            # Check ratio for c_new
            r_new, _ = _compute_ratio_and_grad(c_new, Y, w_theta, w_phi)
            if r_new < best_ratio:
                best_ratio = r_new
                best_coeffs = c_new
                
    # Phase 2: L4 Newton polish on best solution
    if time.time() < deadline - 90:
        best_coeffs, _ = _projected_newton_step(best_coeffs, Y, w_theta, w_phi, max_iter=200, tol=1e-13)
        best_ratio = _compute_ratio_and_grad(best_coeffs, Y, w_theta, w_phi)[0]
        
    # Phase 3: More random restarts
    for i in range(25):
        if time.time() > deadline - 60:
            break
            
        if i % 3 == 0:
            s = np.random.randn(21)
            s = s / np.linalg.norm(s)
        elif i % 3 == 1:
            s = best_coeffs + 0.3 * np.random.randn(21)
            s = s / np.linalg.norm(s)
        else:
            s = np.zeros(21)
            for m in range(np.random.randint(1, 6)):
                if m == 0:
                    s[10] = 1.0
                else:
                    s[10 + m] = np.random.choice([-1, 1])
                    s[10 - m] = np.random.choice([-1, 1])
            s = s / np.linalg.norm(s)
            
        try:
            result = minimize(
                lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0],
                s,
                jac=lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1],
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x
        except Exception:
            pass
            
        if time.time() < deadline - 45:
            c_new, _ = _projected_newton_step(best_coeffs, Y, w_theta, w_phi, max_iter=30, tol=1e-13)
            r_new, _ = _compute_ratio_and_grad(c_new, Y, w_theta, w_phi)
            if r_new < best_ratio:
                best_ratio = r_new
                best_coeffs = c_new
                
    # Phase 4: High-precision final polish
    if time.time() < deadline - 30:
        # Fixed point iteration
        c_fp = _fixed_point_polish(best_coeffs, Y, w_theta, w_phi, max_iter=200, tol=1e-15)
        r_fp, _ = _compute_ratio_and_grad(c_fp, Y, w_theta, w_phi)
        if r_fp < best_ratio:
            best_ratio = r_fp
            best_coeffs = c_fp
            
        # Final Newton polish
        best_coeffs, _ = _projected_newton_step(best_coeffs, Y, w_theta, w_phi, max_iter=100, tol=1e-15)
        best_ratio = _compute_ratio_and_grad(best_coeffs, Y, w_theta, w_phi)[0]
        
    # Phase 5: Final check
    if time.time() < deadline - 15:
        for perturb_scale in [0.1, 0.2]:
            if time.time() > deadline - 15:
                break
            s = best_coeffs + perturb_scale * np.random.randn(21)
            s = s / np.linalg.norm(s)
            try:
                result = minimize(
                    lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0],
                    s,
                    jac=lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 40}
                )
                if result.fun < best_ratio:
                    best_ratio = result.fun
                    best_coeffs = result.x
            except Exception:
                pass
                
    # Final validation
    best_coeffs = best_coeffs / np.linalg.norm(best_coeffs)
    
    if len(best_coeffs) != 21:
        return x0
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return x0
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return x0
        
    return best_coeffs

# EVOLVE-BLOCK-END
