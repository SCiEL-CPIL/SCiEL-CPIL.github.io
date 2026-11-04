"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
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


def _hermite_poly_scalar(n, x):
    """Compute physicist's Hermite polynomial Hn(x) for scalar x."""
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    h0 = 1.0
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    start_time = time.time()
    deadline = 175  # seconds, keep margin for evaluator kill at 210s

    # Basis functions with g(0)=0 constraint baked in
    # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # c12 = -(c0 + 12*c4 + 1680*c8) / 40320
    # v_i(x) = (H_i(x) - coeff_i * H12(x) / 40320) * exp(-x^2/2)
    # where coeff_0=1, coeff_4=12, coeff_8=1680

    def v_func(i, x):
        """Compute basis function v_i(x) for scalar x."""
        if i == 0:
            return (1.0 - _hermite_poly_scalar(12, x) / 40320.0) * np.exp(-x**2 / 2.0)
        elif i == 1:
            return (_hermite_poly_scalar(4, x) - 12.0 * _hermite_poly_scalar(12, x) / 40320.0) * np.exp(-x**2 / 2.0)
        elif i == 2:
            return (_hermite_poly_scalar(8, x) - 1680.0 * _hermite_poly_scalar(12, x) / 40320.0) * np.exp(-x**2 / 2.0)

    # Precompute M and Mx using adaptive quadrature
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))

    quad_opts = {'epsabs': 1e-12, 'epsrel': 1e-12, 'limit': 300}

    for i in range(3):
        for j in range(i, 3):
            def integrand_M(x, i=i, j=j):
                return v_func(i, x) * v_func(j, x)

            def integrand_Mx(x, i=i, j=j):
                return x**2 * v_func(i, x) * v_func(j, x)

            Mij, _ = quad(integrand_M, -np.inf, np.inf, **quad_opts)
            Mxij, _ = quad(integrand_Mx, -np.inf, np.inf, **quad_opts)

            M[i, j] = Mij
            M[j, i] = Mij
            Mx[i, j] = Mxij
            Mx[j, i] = Mxij

    def compute_bound_from_coeffs(c):
        """Compute bound for a specific coefficient vector c."""
        c = np.array(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        if not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
        return x2_norm / norm2

    def compute_bound_spherical(theta, phi):
        """Compute bound using spherical coordinates on the unit sphere."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        return compute_bound_from_coeffs(c)

    # Initial seed coefficients
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_bound = compute_bound_from_coeffs(c_seed)
    best_coeffs = c_seed.copy()

    # Normalize seed for spherical search
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
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
                maxiter=200,
                tol=1e-12,
                popsize=30,
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
        rng = np.random.RandomState(123)

        start_points = [(theta0, phi0)]

        # Golden angle spiral
        golden_angle = np.pi * (3 - np.sqrt(5))
        n_spiral = 40
        for i in range(n_spiral):
            z = 1 - 2 * (i + 0.5) / n_spiral
            theta = np.arccos(np.clip(z, -1.0, 1.0))
            phi = golden_angle * i
            if phi < 0:
                phi += 2 * np.pi
            start_points.append((theta, phi))

        # Random points
        for _ in range(30):
            theta = np.arccos(2 * rng.random() - 1)
            phi = 2 * np.pi * rng.random()
            start_points.append((theta, phi))

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
                    options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 10000, 'adaptive': True}
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

            for polish_iter in range(5):
                tol_factor = 10 ** (-15 - polish_iter)
                polish_res = minimize(
                    compute_bound_spherical,
                    x0=np.array([theta_best, phi_best]),
                    method='Nelder-Mead',
                    options={'xatol': tol_factor, 'fatol': tol_factor / 100, 'maxiter': 50000, 'adaptive': True}
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
