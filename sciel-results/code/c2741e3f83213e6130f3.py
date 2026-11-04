"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time
import math

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    # Hermite polynomial values at 0 (Physicists' Hermite)
    h0_0 = eval_hermite(0, 0)
    h4_0 = eval_hermite(4, 0)
    h8_0 = eval_hermite(8, 0)
    h12_0 = eval_hermite(12, 0)

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    # Precompute grid
    x = np.linspace(-12, 12, 40000)
    dx = x[1] - x[0]

    # Precompute Hermite polynomials on grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)

    # Weights for exact L2 norm calculation: W_i = sqrt(pi) * 2^i * i!
    W0 = math.sqrt(math.pi) * 2**0 * math.factorial(0)
    W4 = math.sqrt(math.pi) * 2**4 * math.factorial(4)
    W8 = math.sqrt(math.pi) * 2**8 * math.factorial(8)
    W12 = math.sqrt(math.pi) * 2**12 * math.factorial(12)

    def compute_bound(c0, c4, c8):
        c12 = get_c12(c0, c4, c8)

        # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
        poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        g = poly * gauss

        # L2 norm squared (Exact)
        norm_sq = c0**2 * W0 + c4**2 * W4 + c8**2 * W8 + c12**2 * W12
        if norm_sq < 1e-15:
            return 1e10

        # L1 norm of g (Numerical)
        l1_g = np.sum(np.abs(g)) * dx

        # Uncertainty product bound ~ (L1 / sqrt(L2))^2 = L1^2 / L2
        return (l1_g * l1_g) / norm_sq

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0

    best_score = compute_bound(c0_init, c4_init, c8_init)
    best_coeffs = [c0_init, c4_init, c8_init]

    # Strategy 1: Differential Evolution on the sphere (2D parameterization)
    # The objective is homogeneous of degree 0 in (c0, c4, c8), so we can
    # parameterize by direction on the unit sphere in R^3.
    # Use spherical coordinates: c0 = r*cos(theta)*cos(phi), c4 = r*cos(theta)*sin(phi), c8 = r*sin(theta)
    # Since the objective is scale-invariant, set r=1.
    
    def bound_from_angles(theta, phi):
        c0 = np.cos(theta) * np.cos(phi)
        c4 = np.cos(theta) * np.sin(phi)
        c8 = np.sin(theta)
        return compute_bound(c0, c4, c8)

    # Run differential evolution on (theta, phi) in [0, pi] x [0, 2*pi]
    if time.time() < deadline - 60.0:
        try:
            de_bounds = [(0, np.pi), (0, 2*np.pi)]
            np.random.seed(42)
            de_result = differential_evolution(
                bound_from_angles,
                bounds=de_bounds,
                maxiter=300,
                tol=1e-10,
                popsize=30,
                seed=42,
                polish=True
            )
            if de_result.fun < best_score:
                theta_opt, phi_opt = de_result.x
                c0_opt = np.cos(theta_opt) * np.cos(phi_opt)
                c4_opt = np.cos(theta_opt) * np.sin(phi_opt)
                c8_opt = np.sin(theta_opt)
                best_score = de_result.fun
                best_coeffs = [c0_opt, c4_opt, c8_opt]
        except Exception:
            pass

    # Strategy 2: Multi-start Powell from the best found
    if time.time() < deadline - 30.0:
        n_starts = 50
        max_iter_per_start = 2000
        np.random.seed(123)
        
        for i in range(n_starts):
            if time.time() > deadline - 15.0:
                break

            if i == 0:
                x0 = best_coeffs[:]
            else:
                # Random initialization on the sphere
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2*np.pi)
                x0 = [
                    np.cos(theta) * np.cos(phi),
                    np.cos(theta) * np.sin(phi),
                    np.sin(theta)
                ]

            try:
                res = minimize(
                    lambda c: compute_bound(*c),
                    x0,
                    method='Powell',
                    options={'maxiter': max_iter_per_start, 'xtol': 1e-10, 'ftol': 1e-12, 'adaptive': True}
                )

                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x.tolist()
            except Exception:
                pass

    # Strategy 3: Polish with L-BFGS-B from the best found
    if time.time() < deadline - 10.0 and best_score < 1e10:
        try:
            res = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.tolist()
        except Exception:
            pass

    # Strategy 4: One more polish with Nelder-Mead for robustness
    if time.time() < deadline - 5.0 and best_score < 1e10:
        try:
            res = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.tolist()
        except Exception:
            pass

    # Final: scale the best coefficients to have unit L2 norm for consistency
    c0_f, c4_f, c8_f = best_coeffs
    c12_f = get_c12(c0_f, c4_f, c8_f)
    norm_sq_f = c0_f**2 * W0 + c4_f**2 * W4 + c8_f**2 * W8 + c12_f**2 * W12
    if norm_sq_f > 1e-15:
        scale = 1.0 / np.sqrt(norm_sq_f)
        best_coeffs = [c0_f * scale, c4_f * scale, c8_f * scale]

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
