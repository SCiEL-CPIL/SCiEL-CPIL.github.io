"""Rewritten program: KKT-based stationary point search for the uncertainty inequality."""
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
    deadline = 175

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

    # Spherical parameterization
    def compute_bound_spherical(theta, phi):
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        return compute_bound_from_coeffs(c)

    # Derive KKT conditions: minimize f(c) = sqrt((c^T Mx c)(c^T Mf c)) / (c^T M c)
    # subject to c^T c = 1 (on the unit sphere)
    # At optimum: grad f = lambda * c
    # This gives a system of equations. Let's derive them.
    #
    # Let A = Mx, B = Mf, C = M
    # f = sqrt((c^T A c)(c^T B c)) / (c^T C c)
    # Let u = c^T A c, v = c^T B c, w = c^T C c
    # f = sqrt(u*v) / w
    #
    # grad f = (1/2) * sqrt(v/u) * (2Ac) * w - sqrt(u*v) * (2Cc) / w^2  ... (for u part)
    #          + (1/2) * sqrt(u/v) * (2Bc) * w - sqrt(u*v) * (2Cc) / w^2  ... (for v part)
    #
    # This is complex. Instead, let's minimize log f = 0.5*log(u) + 0.5*log(v) - log(w)
    # grad log f = A c / u + B c / v - C c / w
    # KKT: A c / u + B c / v - C c / w = mu * c
    #
    # This is a nonlinear system. Let's solve it numerically using fsolve.

    from scipy.optimize import fsolve

    def kkt_equations(c):
        """KKT conditions for the constrained optimization on the unit sphere."""
        c = np.array(c, dtype=float)
        u = c @ Mx @ c
        v = c @ Mf @ c
        w = c @ M @ c
        if u <= 0 or v <= 0 or w <= 0:
            return [1e10, 1e10, 1e10]
        # grad log f - mu * c = 0, plus constraint c^T c = 1
        # We have 4 equations for 4 unknowns (c0, c4, c8, mu)
        # But we only return 3 equations here; mu is implicit.
        # Actually, let's set up 4 equations: 3 from KKT, 1 from normalization.
        # But fsolve needs same number of equations as unknowns.
        # Let's use 3 unknowns (c on sphere) and 3 equations from KKT projected.
        # Simpler: just use the gradient of f on the sphere.
        
        # Use a different approach: minimize f on the sphere using Lagrange multiplier
        # The stationary points satisfy: (Mx/u + Mf/v - M/w) c = mu * c
        # where u = c^T Mx c, v = c^T Mf c, w = c^T M c
        # This is an eigenvalue-like problem but nonlinear.
        
        # Let's just return the gradient of f w.r.t. c, projected onto the sphere
        # The gradient of f = sqrt(uv)/w
        # df/dc_i = (1/2) * sqrt(v/u) * 2(Ac)_i * w - sqrt(uv) * 2(Cc)_i / w^2
        #          + (1/2) * sqrt(u/v) * 2(Bc)_i * w - sqrt(uv) * 2(Cc)_i / w^2
        Ac = Mx @ c
        Bc = Mf @ c
        Cc = M @ c
        grad_u = Ac / u
        grad_v = Bc / v
        grad_w = Cc / w
        # grad log f = grad_u + grad_v - grad_w
        grad_logf = grad_u + grad_v - grad_w
        # Project onto tangent space of sphere: grad_logf - (c^T grad_logf) * c
        proj = grad_logf - np.dot(c, grad_logf) * c
        return proj

    # Find stationary points using fsolve from multiple initial guesses
    best_bound = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_bound = compute_bound_from_coeffs(best_coeffs)

    if time.time() - start_time < deadline - 30:
        # Generate diverse starting points on the sphere
        n_starts = 150
        rng = np.random.RandomState(123)
        
        start_points = []
        # Include seed direction
        c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        c_seed_norm = c_seed / np.linalg.norm(c_seed)
        start_points.append(c_seed_norm)
        
        # Add structured grid points on the sphere
        n_theta = 10
        n_phi = 20
        for i in range(n_theta):
            theta = np.pi * (i + 0.5) / n_theta
            for j in range(n_phi):
                phi = 2 * np.pi * j / n_phi
                start_points.append(np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ]))
        
        # Fill remaining with random points
        n_random = n_starts - len(start_points)
        for _ in range(n_random):
            z = 2 * rng.random() - 1
            theta = 2 * np.pi * rng.random()
            r = np.sqrt(1 - z**2)
            start_points.append(np.array([r * np.cos(theta), r * np.sin(theta), z]))
        
        for c_init in start_points:
            if time.time() - start_time > deadline - 10:
                break
            
            try:
                # Solve KKT equations
                sol, info, ier, msg = fsolve(kkt_equations, c_init, full_output=True, xtol=1e-14)
                if ier == 1:
                    # Normalize solution
                    sol_norm = np.linalg.norm(sol)
                    if sol_norm > 1e-10:
                        sol = sol / sol_norm
                        bound = compute_bound_from_coeffs(sol)
                        if bound < best_bound and np.isfinite(bound):
                            best_bound = bound
                            best_coeffs = sol.copy()
            except Exception:
                pass

    # Final polish with high-precision local optimization
    if time.time() - start_time < deadline - 5:
        try:
            # Convert best_coeffs to spherical coordinates
            c_norm = best_coeffs / np.linalg.norm(best_coeffs)
            theta0 = np.arccos(np.clip(c_norm[2], -1, 1))
            phi0 = np.arctan2(c_norm[1], c_norm[0])
            if phi0 < 0:
                phi0 += 2 * np.pi
            
            res = minimize(
                compute_bound_spherical,
                x0=np.array([theta0, phi0]),
                method='Nelder-Mead',
                options={'xatol': 1e-15, 'fatol': 1e-17, 'maxiter': 5000, 'adaptive': True}
            )
            
            if res.fun < best_bound and np.isfinite(res.fun):
                theta, phi = res.x
                c0 = np.sin(theta) * np.cos(phi)
                c4 = np.sin(theta) * np.sin(phi)
                c8 = np.cos(theta)
                c_candidate = np.array([c0, c4, c8])
                scale = np.linalg.norm(best_coeffs)
                best_coeffs = c_candidate * scale
                best_bound = res.fun
        except Exception:
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
