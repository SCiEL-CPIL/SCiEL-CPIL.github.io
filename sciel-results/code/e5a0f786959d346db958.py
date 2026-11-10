import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
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
    
    return Y, weights_2d

def _objective(coeffs, Y, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    return norm4 / norm2

def _objective_and_grad(coeffs, Y, weights_2d):
    """Compute objective and correct analytic gradient."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    
    # d||f||4/dc_m = (1/norm4^3) * sum_ij(f^3 * Y_m * w_ij)
    # d||f||2/dc_m = (1/norm2) * sum_ij(f * Y_m * w_ij)
    # d(R)/dc_m = (1/norm2)*dnorm4/dc_m - (norm4/norm2^2)*dnorm2/dc_m
    
    f3 = f**3
    f3Y = np.einsum('ij,mij->mij', f3, Y)  # (21, 70, 130)
    fY = np.einsum('ij,mij->mij', f, Y)     # (21, 70, 130)
    
    dnorm4_dc = np.einsum('mij,ij->m', f3Y, w) / (norm4**3)
    dnorm2_dc = np.einsum('mij,ij->m', fY, w) / norm2
    
    grad = dnorm4_dc / norm2 - (norm4 / norm2**2) * dnorm2_dc
    
    return norm4 / norm2, grad

def _objective_sphere(c, Y, weights_2d):
    """Objective on unit sphere: since R is scale-invariant, minimize ||f||_4 when ||c||_2=1."""
    f = np.tensordot(c, Y, axes=(0, 0))
    w = weights_2d
    norm4 = np.sum(f**4 * w)**0.25
    return norm4

def _objective_sphere_grad(c, Y, weights_2d):
    """Gradient of ||f||_4 with respect to c, projected to tangent of unit sphere."""
    f = np.tensordot(c, Y, axes=(0, 0))
    w = weights_2d
    norm4 = np.sum(f**4 * w)**0.25
    if norm4 < 1e-30:
        return 1e10, np.zeros_like(c)
    
    # d||f||4/dc_m = (1/norm4^3) * sum(f^3 * Y_m * w)
    f3Y = np.einsum('ij,mij->mij', f**3, Y)
    dnorm4_dc = np.einsum('mij,ij->m', f3Y, w) / (norm4**3)
    
    # Project to tangent space: grad_t = grad - (grad . c) * c
    dot = np.dot(dnorm4_dc, c)
    grad_t = dnorm4_dc - dot * c
    
    return norm4, grad_t

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s limit
    
    Y, weights_2d = _build_harmonics()
    
    best_coeffs = np.ones(21) / np.sqrt(21)
    best_score = _objective(best_coeffs, Y, weights_2d)
    
    # Structured initial guesses
    structured_guesses = []
    
    # m=0 only
    c = np.zeros(21)
    c[10] = 1.0
    structured_guesses.append(c)
    
    # Even m cos only
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10+m] = 1.0
    structured_guesses.append(c)
    
    # All cos terms
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = 1.0
    structured_guesses.append(c)
    
    # Symmetric: c[m] = c[-m] for all m
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = 1.0
        if m > 0:
            c[10-m] = 1.0
    structured_guesses.append(c)
    
    # Alternating signs on cos terms
    c = np.zeros(21)
    for m in range(0, 11):
        c[10+m] = (-1)**m
    structured_guesses.append(c)
    
    # m=0 dominant with small others
    c = np.zeros(21)
    c[10] = 1.0
    for m in [1, 2, 3, 4, 5]:
        c[10+m] = 0.1
        c[10-m] = 0.1
    structured_guesses.append(c)
    
    for c in structured_guesses:
        s = _objective(c, Y, weights_2d)
        if s < best_score and np.isfinite(s):
            best_score = s
            best_coeffs = c.copy()
    
    # Multi-restart optimization with correct gradient
    np.random.seed(42)
    n_restarts = 40
    
    for i in range(n_restarts):
        if time.time() - start_time > deadline - 60:
            break
        
        # Generate diverse starting points
        if i < 10:
            # Random unit vectors
            x0 = np.random.randn(21)
        elif i < 20:
            # Sparse random
            x0 = np.random.randn(21)
            mask = np.random.random(21) < 0.3
            x0 *= mask
            if np.linalg.norm(x0) < 1e-10:
                x0 = np.random.randn(21)
        elif i < 30:
            # Perturb best solution
            x0 = best_coeffs + 0.5 * np.random.randn(21)
        else:
            # Structured random
            x0 = np.zeros(21)
            for m in range(11):
                if np.random.random() < 0.5:
                    x0[10+m] = np.random.randn()
                if m > 0 and np.random.random() < 0.5:
                    x0[10-m] = np.random.randn()
        
        x0 = x0 / np.linalg.norm(x0) * 5.0
        
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                x0,
                jac=lambda c: _objective_and_grad(c, Y, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Differential evolution for global search
    if time.time() - start_time < deadline - 180:
        try:
            bounds = [(-10, 10)] * 21
            de_res = differential_evolution(
                lambda c: _objective(c, Y, weights_2d),
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            if de_res.fun < best_score and np.isfinite(de_res.fun):
                best_score = de_res.fun
                best_coeffs = de_res.x.copy()
        except Exception:
            pass
    
    # Final polish with high-precision L-BFGS-B using analytic gradient
    if time.time() - start_time < deadline - 30:
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                best_coeffs,
                jac=lambda c: _objective_and_grad(c, Y, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Second polish pass with even tighter tolerance
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                best_coeffs,
                jac=lambda c: _objective_and_grad(c, Y, weights_2d)[1],
                method='L-BFGS-B',
                options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-14}
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
        # Scale to valid range
        if max_abs < 1e-3:
            best_coeffs *= 1.0 / max_abs if max_abs > 0 else 1.0
        elif max_abs > 1e5:
            best_coeffs *= 1e5 / max_abs
        if not np.all(np.isfinite(best_coeffs)):
            return np.ones(21)
    
    return best_coeffs.astype(np.float64)
