"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, fsolve
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
        c = np.array(c)
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

    def compute_bound_from_spherical(theta, phi):
        """Compute bound for spherical parameters."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    # Initial best guess from seed
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    # Convert to spherical for initialization
    c0, c4, c8 = c_seed_norm
    theta_init = np.arccos(c8)
    phi_init = np.arctan2(c4, c0)
    if phi_init < 0:
        phi_init += 2 * np.pi

    best_bound = compute_bound_from_spherical(theta_init, phi_init)
    best_theta = theta_init
    best_phi = phi_init

    # Global optimization using Differential Evolution
    if time.time() - start_time < deadline - 50:
        try:
            bounds = [(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)]
            de_result = differential_evolution(
                compute_bound_from_spherical, 
                bounds, 
                seed=42, 
                maxiter=200, 
                tol=1e-8,
                popsize=15,
                polish=True
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_theta, best_phi = de_result.x
        except Exception:
            pass

    # Local refinement with Nelder-Mead on the 2D spherical parameters
    if time.time() - start_time < deadline - 20:
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[best_theta, best_phi], 
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 10000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # NEW: Homotopy continuation / fsolve approach to find exact critical points
    # The condition for a critical point is that the gradient of the objective
    # on the sphere is zero. In Cartesian coords, this is equivalent to:
    # (Mx + Mf) @ c = lambda * M @ c  (for some lambda)
    # Actually, the gradient of (c^T Mx c * c^T Mf c) / (c^T M c)^2 is:
    # 2 * (Mx + Mf) @ c / (c^T M c) - 2 * (c^T Mx c * c^T Mf c) / (c^T M c)^3 * 2 * M @ c = 0
    # Let A = Mx + Mf, B = M, f = c^T Mx c, g = c^T Mf c, n = c^T M c
    # grad = 2 A c / n - 4 f g / n^3 B c = 0
    # A c = (2 f g / n^2) B c
    # This is a nonlinear eigenvalue problem.
    
    # We can solve for c using fsolve on the equations derived from the gradient.
    # To avoid the scaling issue, we can fix one coefficient or use a homotopy.
    # Let's define F(c) = A c - mu(c) B c = 0, where mu(c) = 2 * (c^T Mx c) * (c^T Mf c) / (c^T M c)^2
    # This is 3 equations in 3 unknowns. However, the solution is only determined up to scale.
    # We can add a constraint ||c|| = 1 or c0 = 1.
    # Let's use the constraint c0 = 1 and solve for c4, c8.
    # F(c4, c8) = [A c - mu(c) B c] with c = [1, c4, c8]^T, taking the last 2 components.
    
    def _fixed_point_equations(params):
        c4, c8 = params
        c = np.array([1.0, c4, c8])
        
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return [1e6, 1e6]
            
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        
        if x2_norm < 0 or freq2_norm < 0:
            return [1e6, 1e6]

        mu = 2.0 * x2_norm * freq2_norm / (norm2**2)
        
        A = Mx + Mf
        Ac = A @ c
        Bc = M @ c
        
        # We want Ac - mu * Bc = 0
        # We only use the last 2 equations because the first is dependent on the scaling?
        # Actually, if we fix c0=1, we have 2 free variables, so we need 2 equations.
        # The vector equation Ac - mu Bc = 0 has 3 components.
        # Since c is determined up to scale, and we fixed c0=1, the system is overdetermined?
        # No, mu depends on c.
        # Let's just return the last 2 components.
        eqs = Ac - mu * Bc
        return [eqs[1], eqs[2]]

    # Try to solve using fsolve from multiple starting points
    if time.time() - start_time < deadline - 10:
        starting_points = [
            [best_coeffs[1]/best_coeffs[0], best_coeffs[2]/best_coeffs[0]] if best_coeffs[0] != 0 else [0, 0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1e-3, 1e-3],
            [-1e-3, -1e-3],
        ]
        
        # Ensure we have a valid starting point from the previous best
        c_best_cart = np.array([
            np.sin(best_theta) * np.cos(best_phi),
            np.sin(best_theta) * np.sin(best_phi),
            np.cos(best_theta)
        ])
        if abs(c_best_cart[0]) > 1e-8:
            starting_points.insert(0, [c_best_cart[1]/c_best_cart[0], c_best_cart[2]/c_best_cart[0]])

        for sp_ in starting_points:
            if time.time() - start_time > deadline - 5:
                break
            try:
                sol, info, ier, msg = fsolve(_fixed_point_equations, sp_, full_output=True)
                if ier == 1:
                    c4_sol, c8_sol = sol
                    c_sol = np.array([1.0, c4_sol, c8_sol])
                    
                    # Check if this is a valid solution (finite, non-zero norm)
                    if np.all(np.isfinite(c_sol)) and np.linalg.norm(c_sol) > 1e-8:
                        bound = compute_bound_from_coeffs(c_sol)
                        if bound < best_bound:
                            best_bound = bound
                            # Update best_theta, best_phi
                            c_norm = c_sol / np.linalg.norm(c_sol)
                            best_theta = np.arccos(c_norm[2])
                            best_phi = np.arctan2(c_norm[1], c_norm[0])
                            if best_phi < 0:
                                best_phi += 2 * np.pi
            except Exception:
                pass

    # Final polish with higher precision Nelder-Mead
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                compute_bound_from_spherical, 
                x0=[best_theta, best_phi], 
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 50000, 'adaptive': True}
            )
            if res.fun < best_bound:
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
