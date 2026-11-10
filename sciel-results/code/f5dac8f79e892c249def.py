import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def _build_spectral():
    """Precompute the quartic tensor K_{abcd} = ∫ Y_a Y_b Y_c Y_d dΩ."""
    x, w = np.polynomial.legendre.leggauss(70)
    phi = np.linspace(0, 2*np.pi, 130, endpoint=False)
    
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    dphi = 2*np.pi / 130
    weights_2d = np.outer(w, np.ones(130)) * dphi
    
    P = {}
    for m in range(0, 11):
        P[m] = lpmv(m, 10, cos_theta)
    
    N = {}
    for m in range(0, 11):
        N[m] = np.sqrt(21.0/(4*np.pi) * factorial(10-m)/factorial(10+m))
    
    Y = np.zeros((21, 70, 130))
    for idx, m in enumerate(range(-10, 11)):
        if m == 0:
            Y[idx] = N[0] * P[0]
        elif m > 0:
            Y[idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_grid)
        else:
            am = abs(m)
            Y[idx] = np.sqrt(2) * N[am] * P[am] * np.sin(am * phi_grid)
    
    # Flatten the grid for tensor computation
    Y_flat = Y.reshape(21, -1)  # (21, 9100)
    w_flat = weights_2d.ravel()  # (9100,)
    
    # Compute K_{abcd} = sum_i Y_a(i) Y_b(i) Y_c(i) Y_d(i) w(i)
    # Use einsum for efficiency
    # K[a,b,c,d] = sum_i Y[a,i] * Y[b,i] * Y[c,i] * Y[d,i] * w[i]
    # This is expensive with full einsum. Let's use a smarter approach.
    
    # We can compute it as: for each pair (a,b), compute T_ab = sum_i Y[a,i]*Y[b,i]*w[i]*Y[c,i]*Y[d,i]
    # Actually, let's just do it with broadcasting in chunks or use the fact that
    # K is symmetric.
    
    # Better approach: compute M = Y_flat * sqrt(w_flat)[:,None]  (21, 9100)
    # Then K[a,b,c,d] = M[a,:] @ (M[b,:] * M[c,:] * M[d,:])
    # Still expensive. Let's use a different strategy.
    
    # Since we only need to evaluate Q(c) = c^T K c (quartic) and its gradient,
    # we can compute Q(c) = sum_i (sum_a c_a Y[a,i])^4 w[i]
    # and gradient_m = (1/Q^(3/4)) * sum_i 4 (sum_a c_a Y[a,i])^3 Y[m,i] w[i]
    
    # So we don't actually need to precompute K! We just need Y_flat and w_flat.
    # The spectral insight is that ||f||_2 = ||c||_2 (no grid needed for L2),
    # and we can compute L4 via the grid but the optimization is in 21D.
    
    return Y_flat, w_flat

def _objective_spectral(c, Y_flat, w_flat):
    """Compute R = ||f||_4 / ||f||_2 using spectral form."""
    f = Y_flat @ c  # (9100,)
    Q4 = np.sum(f**4 * w_flat)
    if Q4 < 1e-30:
        return 1e10
    norm4 = Q4**0.25
    norm2 = np.linalg.norm(c)
    if norm2 < 1e-30:
        return 1e10
    return norm4 / norm2

def _objective_spectral_grad(c, Y_flat, w_flat):
    """Compute objective and gradient for R = ||f||_4 / ||f||_2."""
    f = Y_flat @ c  # (9100,)
    Q4 = np.sum(f**4 * w_flat)
    if Q4 < 1e-30:
        return 1e10, np.zeros_like(c)
    norm4 = Q4**0.25
    norm2 = np.linalg.norm(c)
    if norm2 < 1e-30:
        return 1e10, np.zeros_like(c)
    
    # d||f||_4/dc_m = (1/norm4^3) * sum_i f(i)^3 Y[m,i] w[i]
    f3 = f**3
    dnorm4_dc = Y_flat.T @ (f3 * w_flat) / (norm4**3)
    
    # d||f||_2/dc_m = c_m / norm2
    dnorm2_dc = c / norm2
    
    # R = norm4 / norm2
    # dR/dc_m = dnorm4_dc_m / norm2 - norm4/norm2^2 * dnorm2_dc_m
    grad = dnorm4_dc / norm2 - (norm4 / norm2**2) * dnorm2_dc
    
    return norm4 / norm2, grad

def _sphere_objective(c, Y_flat, w_flat):
    """On unit sphere, minimize ||f||_4 since ||f||_2 = 1."""
    f = Y_flat @ c
    Q4 = np.sum(f**4 * w_flat)
    return Q4**0.25

def _sphere_grad(c, Y_flat, w_flat):
    f = Y_flat @ c
    Q4 = np.sum(f**4 * w_flat)
    if Q4 < 1e-30:
        return 1e10, np.zeros_like(c)
    norm4 = Q4**0.25
    
    f3 = f**3
    dnorm4_dc = Y_flat.T @ (f3 * w_flat) / (norm4**3)
    
    # Project to tangent space
    dot = np.dot(dnorm4_dc, c)
    grad_t = dnorm4_dc - dot * c
    
    return norm4, grad_t

def _sphere_polish(c, Y_flat, w_flat, max_iter=5000, tol=1e-14):
    """Sphere-constrained gradient descent with backtracking."""
    c = c.copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return c
    c = c / nrm
    
    prev_val = _sphere_objective(c, Y_flat, w_flat)
    lr = 0.01
    
    for it in range(max_iter):
        val, grad_t = _sphere_grad(c, Y_flat, w_flat)
        if abs(prev_val - val) < tol:
            break
        prev_val = val
        
        step = lr
        improved = False
        for _ in range(30):
            c_new = c - step * grad_t
            c_new = c_new / np.linalg.norm(c_new)
            val_new = _sphere_objective(c_new, Y_flat, w_flat)
            if val_new < val:
                c = c_new
                improved = True
                break
            step *= 0.5
        if not improved:
            break
        if val < prev_val:
            lr = min(lr * 1.2, 0.1)
        else:
            lr *= 0.5
    
    return c

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0
    
    Y_flat, w_flat = _build_spectral()
    
    best_coeffs = np.ones(21) / np.sqrt(21)
    best_score = _objective_spectral(best_coeffs, Y_flat, w_flat)
    
    # Structured initial guesses
    structured_guesses = []
    
    c = np.zeros(21); c[10] = 1.0
    structured_guesses.append(c)
    
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10+m] = 1.0
    structured_guesses.append(c)
    
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = 1.0
    structured_guesses.append(c)
    
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = 1.0
        if m > 0:
            c[10-m] = 1.0
    structured_guesses.append(c)
    
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = (-1)**m
    structured_guesses.append(c)
    
    c = np.zeros(21)
    c[10] = 1.0
    for m in [1, 2, 3, 4, 5]:
        c[10+m] = 0.1
        c[10-m] = 0.1
    structured_guesses.append(c)
    
    c = np.zeros(21)
    c[10] = 1.0
    c[11] = 0.8; c[9] = 0.8
    c[12] = 0.6; c[8] = 0.6
    structured_guesses.append(c)
    
    c = np.zeros(21)
    for m in [8, 9, 10]:
        c[10+m] = 1.0
        if m > 0:
            c[10-m] = 1.0
    structured_guesses.append(c)
    
    for c in structured_guesses:
        s = _objective_spectral(c, Y_flat, w_flat)
        if s < best_score and np.isfinite(s):
            best_score = s
            best_coeffs = c.copy()
    
    # Multi-restart L-BFGS-B optimization
    np.random.seed(42)
    n_restarts = 100
    
    for i in range(n_restarts):
        if time.time() - start_time > deadline - 120:
            break
        
        if i < 20:
            x0 = np.random.randn(21)
        elif i < 40:
            sparsity = 0.2 + 0.6 * (i - 20) / 20
            x0 = np.random.randn(21)
            mask = np.random.random(21) < sparsity
            x0 *= mask
            if np.linalg.norm(x0) < 1e-10:
                x0 = np.random.randn(21)
        elif i < 60:
            noise_scale = 0.3 + 0.7 * (i - 40) / 20
            x0 = best_coeffs + noise_scale * np.random.randn(21)
        elif i < 80:
            x0 = np.zeros(21)
            for m in range(11):
                if np.random.random() < 0.5:
                    val = np.random.randn()
                    x0[10+m] = val
                    if m > 0:
                        x0[10-m] = val * np.random.choice([-1, 1])
        else:
            x0 = np.zeros(21)
            for m in range(11):
                weight = 1.0 / (1.0 + 0.5*m)
                if np.random.random() < 0.7:
                    x0[10+m] = np.random.randn() * weight
                    if m > 0:
                        x0[10-m] = np.random.randn() * weight
        
        nrm = np.linalg.norm(x0)
        if nrm < 1e-10:
            x0 = np.random.randn(21)
            nrm = np.linalg.norm(x0)
        x0 = x0 / nrm * 5.0
        
        try:
            res = minimize(
                lambda c: _objective_spectral(c, Y_flat, w_flat),
                x0,
                jac=lambda c: _objective_spectral_grad(c, Y_flat, w_flat)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Sphere-constrained polish
    if time.time() - start_time < deadline - 60:
        try:
            polished = _sphere_polish(best_coeffs, Y_flat, w_flat, max_iter=5000, tol=1e-14)
            s_polished = _objective_spectral(polished, Y_flat, w_flat)
            if s_polished < best_score and np.isfinite(s_polished):
                best_score = s_polished
                best_coeffs = polished.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() - start_time < deadline - 20:
        try:
            res = minimize(
                lambda c: _objective_spectral(c, Y_flat, w_flat),
                best_coeffs,
                jac=lambda c: _objective_spectral_grad(c, Y_flat, w_flat)[1],
                method='L-BFGS-B',
                options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Second polish
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                lambda c: _objective_spectral(c, Y_flat, w_flat),
                best_coeffs,
                jac=lambda c: _objective_spectral_grad(c, Y_flat, w_flat)[1],
                method='L-BFGS-B',
                options={'maxiter': 100000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Validate
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs < 1e-3:
            best_coeffs *= 1.0 / max_abs if max_abs > 0 else 1.0
        elif max_abs > 1e5:
            best_coeffs *= 1e5 / max_abs
        if not np.all(np.isfinite(best_coeffs)):
            return np.ones(21)
    
    return best_coeffs.astype(np.float64)
