"""L-BFGS-B on sphere with structured grid + random restarts + high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


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
    deadline = 180  # seconds, keep margin for evaluator kill at 210s

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

    # Precompute Gram matrices
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
        """Compute bound for a specific coefficient vector c."""
        c = np.array(c)
        norm2 = c @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        c_norm = c / np.sqrt(norm2)
        x2_norm = c_norm @ Mx @ c_norm
        freq2_norm = c_norm @ Mf @ c_norm
        
        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
            
        delta_x = np.sqrt(x2_norm)
        delta_xi = np.sqrt(freq2_norm)
        return delta_x * delta_xi

    def compute_bound_from_spherical(theta, phi):
        """Compute bound for spherical parameters."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    def objective_sphere(theta, phi):
        """Objective for scipy minimize in spherical coordinates."""
        return compute_bound_from_spherical(theta, phi)

    # Initial best guess from seed
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    c0, c4, c8 = c_seed_norm
    theta_init = np.arccos(np.clip(c8, -1, 1))
    phi_init = np.arctan2(c4, c0)
    if phi_init < 0:
        phi_init += 2 * np.pi

    best_bound = compute_bound_from_spherical(theta_init, phi_init)
    best_theta = theta_init
    best_phi = phi_init

    # Phase 1: Structured grid search to find promising regions
    if time.time() - start_time < deadline - 60:
        # Grid in theta (0 to pi) and phi (0 to 2pi)
        n_theta = 30
        n_phi = 60
        theta_grid = np.linspace(0.05, np.pi - 0.05, n_theta)
        phi_grid = np.linspace(0.05, 2 * np.pi - 0.05, n_phi)
        
        best_grid_bound = np.inf
        best_grid_theta = theta_init
        best_grid_phi = phi_init
        
        for t in theta_grid:
            if time.time() - start_time >= deadline - 55:
                break
            for p in phi_grid:
                b = compute_bound_from_spherical(t, p)
                if b < best_grid_bound:
                    best_grid_bound = b
                    best_grid_theta = t
                    best_grid_phi = p
        
        if best_grid_bound < best_bound:
            best_bound = best_grid_bound
            best_theta = best_grid_theta
            best_phi = best_grid_phi

    # Phase 2: L-BFGS-B local optimization from multiple starting points
    if time.time() - start_time < deadline - 30:
        rng = np.random.RandomState(42)
        
        # Collect starting points: seed, grid best, and random
        start_points = [(theta_init, phi_init), (best_theta, best_phi)]
        
        # Add random starting points
        n_random = 15
        for _ in range(n_random):
            t_rand = rng.uniform(0.1, np.pi - 0.1)
            p_rand = rng.uniform(0.1, 2 * np.pi - 0.1)
            start_points.append((t_rand, p_rand))
        
        # Also add points near the grid best
        for dt in [-0.1, 0.1, -0.2, 0.2]:
            for dp in [-0.1, 0.1, -0.2, 0.2]:
                t_new = best_theta + dt
                p_new = best_phi + dp
                if 0.05 <= t_new <= np.pi - 0.05:
                    p_new = p_new % (2 * np.pi)
                    if 0.05 <= p_new <= 2 * np.pi - 0.05:
                        start_points.append((t_new, p_new))
        
        for t0, p0 in start_points:
            if time.time() - start_time >= deadline - 25:
                break
            try:
                res = minimize(
                    objective_sphere,
                    x0=[t0, p0],
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
                )
                if res.fun < best_bound and np.isfinite(res.fun):
                    best_bound = res.fun
                    best_theta, best_phi = res.x
            except Exception:
                pass

    # Phase 3: High-precision polish from the best point
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                objective_sphere,
                x0=[best_theta, best_phi],
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 20000}
            )
            if res.fun < best_bound and np.isfinite(res.fun):
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # Phase 4: Nelder-Mead polish with very tight tolerance
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                objective_sphere,
                x0=[best_theta, best_phi],
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 20000, 'adaptive': True}
            )
            if res.fun < best_bound and np.isfinite(res.fun):
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # Convert back to Cartesian coefficients
    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)
    
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
