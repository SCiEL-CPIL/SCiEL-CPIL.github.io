"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    deadline = 195  # seconds, keep margin for evaluator kill at 210s

    # Setup grid
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

    # Precompute Gram matrices
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
        """Compute bound for a specific coefficient vector c."""
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

    def compute_bound_and_grad_spherical(theta, phi):
        """Compute bound and analytic gradient for spherical parameters."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf, np.array([0.0, 0.0])
        
        x2_val = c @ Mx @ c
        f2_val = c @ Mf @ c
        
        if not np.isfinite(x2_val) or x2_val < 0 or not np.isfinite(f2_val) or f2_val < 0:
            return np.inf, np.array([0.0, 0.0])
        
        # f = sqrt(x2_val * f2_val) / norm2
        # log f = 0.5*log(x2_val) + 0.5*log(f2_val) - log(norm2)
        # grad_log_f = 0.5*grad_x2/x2_val + 0.5*grad_f2/f2_val - grad_norm2/norm2
        
        # Derivatives w.r.t. c:
        # d(norm2)/dc = 2*M*c
        # d(x2_val)/dc = 2*Mx*c
        # d(f2_val)/dc = 2*Mf*c
        
        # d(theta, phi) -> dc/dtheta, dc/dphi
        dc_dtheta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
        dc_dphi = np.array([-np.sin(theta)*np.sin(phi), np.sin(theta)*np.cos(phi), 0.0])
        
        # Gradient of each quadratic form w.r.t. theta and phi
        dnorm2_dtheta = 2 * (M @ c) @ dc_dtheta
        dnorm2_dphi = 2 * (M @ c) @ dc_dphi
        dx2_dtheta = 2 * (Mx @ c) @ dc_dtheta
        dx2_dphi = 2 * (Mx @ c) @ dc_dphi
        df2_dtheta = 2 * (Mf @ c) @ dc_dtheta
        df2_dphi = 2 * (Mf @ c) @ dc_dphi
        
        # Gradient of log f
        dlogf_dtheta = 0.5 * dx2_dtheta / x2_val + 0.5 * df2_dtheta / f2_val - dnorm2_dtheta / norm2
        dlogf_dphi = 0.5 * dx2_dphi / x2_val + 0.5 * df2_dphi / f2_val - dnorm2_dphi / norm2
        
        # Gradient of f = f * gradient of log f
        f_val = np.sqrt(x2_val * f2_val) / norm2
        grad_f = np.array([f_val * dlogf_dtheta, f_val * dlogf_dphi])
        
        return f_val, grad_f

    def compute_bound_from_spherical(theta, phi):
        """Compute bound for spherical parameters."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    # Initial best guess from seed
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    c0, c4, c8 = c_seed_norm
    theta_init = np.arccos(np.clip(c8, -0.999999, 0.999999))
    phi_init = np.arctan2(c4, c0)
    if phi_init < 0:
        phi_init += 2 * np.pi

    best_bound = compute_bound_from_spherical(theta_init, phi_init)
    best_theta = theta_init
    best_phi = phi_init

    # Generate diverse starting points
    rng = np.random.RandomState(42)
    
    # Add seed and structured points
    starts = [(theta_init, phi_init)]
    
    # Random points
    for _ in range(30):
        t = rng.uniform(0.01, np.pi - 0.01)
        p = rng.uniform(0.01, 2 * np.pi - 0.01)
        starts.append((t, p))
    
    # Structured grid points
    for t in np.linspace(0.2, np.pi - 0.2, 15):
        for p in np.linspace(0.1, 2 * np.pi - 0.1, 15):
            starts.append((t, p))

    # Evaluate all starting points and keep top candidates
    start_scores = []
    for t, p in starts:
        b = compute_bound_from_spherical(t, p)
        if np.isfinite(b) and b < best_bound:
            best_bound = b
            best_theta, best_phi = t, p
        start_scores.append((b, t, p))
    
    # Sort by score and take top 20 for local optimization
    start_scores.sort(key=lambda x: x[0] if np.isfinite(x[0]) else np.inf)
    top_starts = [(t, p) for _, t, p in start_scores[:20] if np.isfinite(compute_bound_from_spherical(t, p))]
    
    # Always include the current best
    if (best_theta, best_phi) not in top_starts:
        top_starts.append((best_theta, best_phi))
    
    # Local refinement with Nelder-Mead from top starting points
    for t, p in top_starts:
        if time.time() - start_time >= deadline - 30:
            break
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[t, p], 
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 30000, 'adaptive': True, 'maxfev': 50000}
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    # Gradient-based refinement using BFGS with analytic gradient
    if time.time() - start_time < deadline - 15:
        try:
            res = minimize(
                lambda x: compute_bound_and_grad_spherical(x[0], x[1])[0],
                x0=[best_theta, best_phi],
                jac=lambda x: compute_bound_and_grad_spherical(x[0], x[1])[1],
                method='BFGS',
                bounds=[(0.001, np.pi - 0.001), (0.001, 2 * np.pi - 0.001)],
                options={'maxiter': 5000, 'gtol': 1e-14, 'disp': False}
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    # Final high-precision Newton-like polish using L-BFGS-B with tight tolerance
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                lambda x: compute_bound_and_grad_spherical(x[0], x[1])[0],
                x0=[best_theta, best_phi],
                jac=lambda x: compute_bound_and_grad_spherical(x[0], x[1])[1],
                method='L-BFGS-B',
                bounds=[(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)],
                options={'maxiter': 10000, 'gtol': 1e-16, 'ftol': 1e-18, 'disp': False}
            )
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
        except Exception:
            pass

    # Additional polish with perturbed initial point
    if time.time() - start_time < deadline - 2:
        try:
            perturbed = [best_theta + 1e-8, best_phi + 1e-8]
            res2 = minimize(
                lambda x: compute_bound_and_grad_spherical(x[0], x[1])[0],
                x0=perturbed,
                jac=lambda x: compute_bound_and_grad_spherical(x[0], x[1])[1],
                method='L-BFGS-B',
                bounds=[(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)],
                options={'maxiter': 5000, 'gtol': 1e-16, 'ftol': 1e-18, 'disp': False}
            )
            if np.isfinite(res2.fun) and res2.fun < best_bound:
                best_bound = res2.fun
                best_theta, best_phi = res2.x[0], res2.x[1]
        except Exception:
            pass

    # Convert back to Cartesian coefficients
    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)
    
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
