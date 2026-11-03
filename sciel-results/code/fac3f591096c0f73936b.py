"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    # Hermite polynomial values at 0
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 9240.0

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    # Precompute grid
    x = np.linspace(-12, 12, 2000)
    dx = x[1] - x[0]

    # Precompute Hermite polynomials on grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)

    def compute_bound(c0, c4, c8):
        if time.time() > deadline:
            return 1e10
        c12 = get_c12(c0, c4, c8)

        # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
        poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        g = poly * gauss

        # L2 norm squared
        norm_sq = np.sum(g**2) * dx
        if norm_sq < 1e-15:
            return 1e10

        # Normalize to L2 = 1
        g_norm = g / np.sqrt(norm_sq)

        # L1 norm of g
        l1_g = np.sum(np.abs(g_norm)) * dx

        # For even functions with n ≡ 0 mod 4, Fourier transform has same form
        # ||ĝ||₁ = ||g||₁ (by symmetry)
        # Uncertainty product bound ~ l1_g * l1_g = l1_g^2
        return l1_g * l1_g

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0

    best_score = compute_bound(c0_init, c4_init, c8_init)
    best_coeffs = [c0_init, c4_init, c8_init]

    # Phase 1: Differential Evolution in spherical coordinates to find global minimum
    # Use angles (theta, phi) and radius r to avoid scale issues
    # c0 = r * sin(theta) * cos(phi)
    # c4 = r * sin(theta) * sin(phi)
    # c8 = r * cos(theta)
    
    def spherical_to_cartesian(params):
        r, theta, phi = params
        c0 = r * np.sin(theta) * np.cos(phi)
        c4 = r * np.sin(theta) * np.sin(phi)
        c8 = r * np.cos(theta)
        return c0, c4, c8
    
    def objective_spherical(params):
        if time.time() > deadline:
            return 1e10
        c0, c4, c8 = spherical_to_cartesian(params)
        return compute_bound(c0, c4, c8)
    
    # Bounds for spherical coordinates
    # r should be positive, theta in [0, pi], phi in [0, 2*pi]
    bounds_spherical = [(0.01, 2.0), (0.0, np.pi), (0.0, 2.0 * np.pi)]
    
    try:
        de_result = differential_evolution(
            objective_spherical,
            bounds_spherical,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=True,
            init='sobol'
        )
        
        if de_result.fun < best_score:
            c0, c4, c8 = spherical_to_cartesian(de_result.x)
            best_score = de_result.fun
            best_coeffs = [c0, c4, c8]
    except Exception:
        pass

    # Phase 2: Multi-start Nelder-Mead for local refinement
    np.random.seed(42)
    n_starts = 30
    max_iter_per_start = 200

    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break

        # Generate initial guess
        if i == 0:
            x0 = best_coeffs
        elif i == 1:
            x0 = [c0_init, c4_init, c8_init]
        else:
            scale = 0.3 * (1 + i * 0.05)
            x0 = [
                best_coeffs[0] + np.random.randn() * scale,
                best_coeffs[1] + np.random.randn() * scale * 0.1,
                best_coeffs[2] + np.random.randn() * scale * 0.01
            ]

        try:
            res = minimize(
                lambda c: compute_bound(*c),
                x0,
                method='Nelder-Mead',
                options={'maxiter': max_iter_per_start, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )

            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.tolist()

        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0 and best_score < 1e10:
        try:
            res = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.tolist()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
