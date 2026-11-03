"""Optimized Fourier-invariant Hermite-Gaussian test function tuning."""
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    start_time = time.time()
    deadline = 170  # seconds, keep margin for evaluator kill at 210s

    # Setup grid - use larger grid for better accuracy
    N = 8192
    x = np.linspace(-25, 25, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Precompute Hermite polynomials
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)

    # Effective basis vectors incorporating the g(0)=0 constraint
    # c12 = -(c0 + 12c4 + 1680c8) / 40320
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    # Precompute Gram matrices for fast evaluation
    M = np.zeros((3, 3))
    M[0, 0] = np.sum(v0**2) * dx
    M[1, 1] = np.sum(v4**2) * dx
    M[2, 2] = np.sum(v8**2) * dx
    M[0, 1] = M[1, 0] = np.sum(v0 * v4) * dx
    M[0, 2] = M[2, 0] = np.sum(v0 * v8) * dx
    M[1, 2] = M[2, 1] = np.sum(v4 * v8) * dx

    Mx = np.zeros((3, 3))
    Mx[0, 0] = np.sum(x**2 * v0**2) * dx
    Mx[1, 1] = np.sum(x**2 * v4**2) * dx
    Mx[2, 2] = np.sum(x**2 * v8**2) * dx
    Mx[0, 1] = Mx[1, 0] = np.sum(x**2 * v0 * v4) * dx
    Mx[0, 2] = Mx[2, 0] = np.sum(x**2 * v0 * v8) * dx
    Mx[1, 2] = Mx[2, 1] = np.sum(x**2 * v4 * v8) * dx

    # Fourier transforms of basis vectors
    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx

    Mf = np.zeros((3, 3))
    Mf[0, 0] = np.sum(freqs**2 * np.abs(G0)**2) * dx
    Mf[1, 1] = np.sum(freqs**2 * np.abs(G4)**2) * dx
    Mf[2, 2] = np.sum(freqs**2 * np.abs(G8)**2) * dx
    Mf[0, 1] = Mf[1, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G4))) * dx
    Mf[0, 2] = Mf[2, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G8))) * dx
    Mf[1, 2] = Mf[2, 1] = np.sum(freqs**2 * np.real(G4 * np.conj(G8))) * dx

    def compute_bound_from_coeffs(c):
        """Compute bound for a specific coefficient vector c using precomputed matrices."""
        c = np.asarray(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        
        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
            
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        return delta_x * delta_xi

    # The objective is scale-invariant: f(c) = f(αc) for any α > 0
    # Use spherical coordinates to fix ||c|| = 1
    def compute_bound_spherical(theta, phi):
        """Compute bound using spherical coordinates on the unit sphere."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        return compute_bound_from_coeffs(c)

    # Initial best guess from seed
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_bound = compute_bound_from_coeffs(c_seed)
    best_coeffs = c_seed.copy()
    
    # Normalize seed to unit sphere for spherical search
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    
    # Convert to spherical coordinates
    theta0 = np.arccos(np.clip(c_seed_norm[2], -1.0, 1.0))
    phi0 = np.arctan2(c_seed_norm[1], c_seed_norm[0])
    if phi0 < 0:
        phi0 += 2 * np.pi

    # Multi-start optimization on the sphere
    # Strategy: Use SLSQP with multiple diverse starting points
    # Also include a gradient-based approach using the analytical gradient
    
    # Compute analytical gradient of the objective w.r.t. (theta, phi)
    def compute_bound_and_grad_spherical(theta, phi):
        """Compute bound and gradient using spherical coordinates."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        
        norm2 = c @ M @ c
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf, np.array([0.0, 0.0])
        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf, np.array([0.0, 0.0])
        
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        val = delta_x * delta_xi
        
        # Gradient computation via chain rule
        # d(norm2)/dc = 2*M*c
        # d(x2_norm)/dc = 2*Mx*c
        # d(freq2_norm)/dc = 2*Mf*c
        
        d_norm2_dc = 2 * M @ c
        d_x2_dc = 2 * Mx @ c
        d_f2_dc = 2 * Mf @ c
        
        # d(delta_x)/dc = (1/(2*delta_x)) * (d_x2_dc/norm2 - x2_norm*d_norm2_dc/norm2^2)
        # = (1/(2*delta_x*norm2^2)) * (norm2*d_x2_dc - x2_norm*d_norm2_dc)
        
        d_delta_x_dc = (norm2 * d_x2_dc - x2_norm * d_norm2_dc) / (2 * delta_x * norm2**2)
        d_delta_xi_dc = (norm2 * d_f2_dc - freq2_norm * d_norm2_dc) / (2 * delta_xi * norm2**2)
        
        # d(val)/dc = delta_xi * d_delta_x_dc + delta_x * d_delta_xi_dc
        d_val_dc = delta_xi * d_delta_x_dc + delta_x * d_delta_xi_dc
        
        # Jacobian of c w.r.t. (theta, phi)
        # c0 = sin(theta)*cos(phi), dc0/dtheta = cos(theta)*cos(phi), dc0/dphi = -sin(theta)*sin(phi)
        # c4 = sin(theta)*sin(phi), dc4/dtheta = cos(theta)*sin(phi), dc4/dphi = sin(theta)*cos(phi)
        # c8 = cos(theta),         dc8/dtheta = -sin(theta),          dc8/dphi = 0
        
        J = np.array([
            [np.cos(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi)],
            [np.cos(theta) * np.sin(phi),  np.sin(theta) * np.cos(phi)],
            [-np.sin(theta),                0.0]
        ])
        
        grad = J.T @ d_val_dc
        return val, grad

    # Multi-start with diverse initial points
    n_starts = 80
    rng = np.random.RandomState(42)
    
    start_points = [(theta0, phi0)]
    
    # Random points on sphere
    for _ in range(n_starts - 1):
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()
        start_points.append((theta, phi))
    
    # Golden angle spiral points for better coverage
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(30):
        z = 1 - 2 * (i + 0.5) / 30
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = golden_angle * i
        if phi < 0:
            phi += 2 * np.pi
        start_points.append((theta, phi))
    
    best_spherical_bound = np.inf
    best_spherical_point = (theta0, phi0)
    
    # Phase 1: SLSQP with analytical gradient from multiple starts
    if time.time() - start_time < deadline - 20:
        for theta_init, phi_init in start_points:
            if time.time() - start_time > deadline - 15:
                break
            
            try:
                res = minimize(
                    compute_bound_and_grad_spherical,
                    x0=np.array([theta_init, phi_init]),
                    method='SLSQP',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-14, 'eps': 1e-8}
                )
                
                if res.fun < best_spherical_bound and np.isfinite(res.fun):
                    best_spherical_bound = res.fun
                    best_spherical_point = res.x.copy()
                    
                    c0 = np.sin(res.x[0]) * np.cos(res.x[1])
                    c4 = np.sin(res.x[0]) * np.sin(res.x[1])
                    c8 = np.cos(res.x[0])
                    c_candidate = np.array([c0, c4, c8])
                    scale = np.linalg.norm(c_seed)
                    best_coeffs = c_candidate * scale
                    
                    if best_spherical_bound < best_bound:
                        best_bound = best_spherical_bound
            except Exception:
                pass
    
    # Phase 2: Nelder-Mead polish from the best point found
    if time.time() - start_time < deadline - 5 and best_spherical_bound < np.inf:
        try:
            polish_res = minimize(
                compute_bound_spherical,
                x0=np.array(best_spherical_point),
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 5000, 'adaptive': True}
            )
            
            if polish_res.fun < best_bound and np.isfinite(polish_res.fun):
                best_bound = polish_res.fun
                theta, phi = polish_res.x
                c0 = np.sin(theta) * np.cos(phi)
                c4 = np.sin(theta) * np.sin(phi)
                c8 = np.cos(theta)
                c_candidate = np.array([c0, c4, c8])
                scale = np.linalg.norm(c_seed)
                best_coeffs = c_candidate * scale
        except Exception:
            pass
    
    # Phase 3: Very tight SLSQP polish with tighter tolerances
    if time.time() - start_time < deadline - 2 and best_spherical_bound < np.inf:
        try:
            theta_best = best_spherical_point[0]
            phi_best = best_spherical_point[1]
            
            final_res = minimize(
                compute_bound_and_grad_spherical,
                x0=np.array([theta_best, phi_best]),
                method='SLSQP',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-16, 'eps': 1e-10}
            )
            
            if final_res.fun < best_bound and np.isfinite(final_res.fun):
                best_bound = final_res.fun
                theta, phi = final_res.x
                c0 = np.sin(theta) * np.cos(phi)
                c4 = np.sin(theta) * np.sin(phi)
                c8 = np.cos(theta)
                c_candidate = np.array([c0, c4, c8])
                scale = np.linalg.norm(c_seed)
                best_coeffs = c_candidate * scale
        except Exception:
            pass

    return best_coeffs
