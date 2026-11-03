"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    deadline = 175  # seconds, keep margin for evaluator kill at 210s

    # Setup grid
    N = 4096
    x = np.linspace(-20, 20, N)
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
        c = np.array(c, dtype=float)
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

    # Phase 1: Global search with differential evolution
    if time.time() - start_time < deadline - 60:
        try:
            bounds = [(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)]
            de_result = differential_evolution(
                compute_bound_spherical,
                bounds,
                seed=42,
                maxiter=150,
                tol=1e-10,
                popsize=20,
                polish=False
            )
            if de_result.fun < best_bound and np.isfinite(de_result.fun):
                best_bound = de_result.fun
                theta_best, phi_best = de_result.x
                c0 = np.sin(theta_best) * np.cos(phi_best)
                c4 = np.sin(theta_best) * np.sin(phi_best)
                c8 = np.cos(theta_best)
                c_candidate = np.array([c0, c4, c8])
                scale = np.linalg.norm(c_seed)
                best_coeffs = c_candidate * scale
        except Exception:
            pass

    # Phase 2: Multi-start Nelder-Mead from diverse points
    if time.time() - start_time < deadline - 20:
        rng = np.random.RandomState(42)
        
        # Generate diverse starting points on the sphere
        start_points = [(theta0, phi0)]
        
        # Golden angle spiral for uniform coverage
        golden_angle = np.pi * (3 - np.sqrt(5))
        n_spiral = 30
        for i in range(n_spiral):
            z = 1 - 2 * (i + 0.5) / n_spiral
            theta = np.arccos(np.clip(z, -1.0, 1.0))
            phi = golden_angle * i
            if phi < 0:
                phi += 2 * np.pi
            start_points.append((theta, phi))
        
        # Random points
        for _ in range(20):
            theta = np.arccos(2 * rng.random() - 1)
            phi = 2 * np.pi * rng.random()
            start_points.append((theta, phi))
        
        # Add the DE result if it improved
        if 'de_result' in dir() and de_result.fun < best_bound:
            start_points.append(tuple(de_result.x))
        
        best_spherical_bound = best_bound
        best_spherical_point = (theta0, phi0)
        
        for theta_init, phi_init in start_points:
            if time.time() - start_time > deadline - 10:
                break
            
            try:
                res = minimize(
                    compute_bound_spherical,
                    x0=np.array([theta_init, phi_init]),
                    method='Nelder-Mead',
                    options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 5000, 'adaptive': True}
                )
                
                if res.fun < best_spherical_bound and np.isfinite(res.fun):
                    best_spherical_bound = res.fun
                    best_spherical_point = res.x
                    
                    c0 = np.sin(res.x[0]) * np.cos(res.x[1])
                    c4 = np.sin(res.x[0]) * np.sin(res.x[1])
                    c8 = np.cos(res.x[0])
                    c_candidate = np.array([c0, c4, c8])
                    scale = np.linalg.norm(c_seed)
                    best_coeffs = c_candidate * scale
                    best_bound = best_spherical_bound
            except Exception:
                pass

    # Phase 3: Ultra-tight final polish
    if time.time() - start_time < deadline - 2:
        try:
            theta_best, phi_best = best_spherical_point
            
            # Multiple polish passes with decreasing tolerance
            for polish_iter in range(3):
                tol_factor = 10 ** (-14 - polish_iter)
                polish_res = minimize(
                    compute_bound_spherical,
                    x0=np.array([theta_best, phi_best]),
                    method='Nelder-Mead',
                    options={'xatol': tol_factor, 'fatol': tol_factor / 100, 'maxiter': 20000, 'adaptive': True}
                )
                
                if polish_res.fun < best_bound and np.isfinite(polish_res.fun):
                    best_bound = polish_res.fun
                    theta_best, phi_best = polish_res.x
                    c0 = np.sin(theta_best) * np.cos(phi_best)
                    c4 = np.sin(theta_best) * np.sin(phi_best)
                    c8 = np.cos(theta_best)
                    c_candidate = np.array([c0, c4, c8])
                    scale = np.linalg.norm(c_seed)
                    best_coeffs = c_candidate * scale
                else:
                    break
        except Exception:
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
