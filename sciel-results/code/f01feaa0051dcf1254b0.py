"""Optimized: vectorized grid-based objective with multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 210s wall-clock limit

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0, matching the seed program's convention."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    # Precompute on a fine grid for fast vectorized objective
    n_grid = 16001
    x_grid = np.linspace(-10, 10, n_grid)
    dx = x_grid[1] - x_grid[0]
    gauss_half = np.exp(-x_grid**2 / 2.0)

    # Precompute Hermite polynomials (physicists' convention as in seed)
    x2 = x_grid**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x8 * x4

    H0_g = np.ones_like(x_grid)
    H4_g = 16.0 * x4 - 48.0 * x2 + 12.0
    H8_g = 256.0 * x8 - 2304.0 * x6 + 5760.0 * x4 - 4320.0 * x2 + 504.0
    H12_g = (4096.0 * x12 - 55296.0 * x10 + 303104.0 * x8
             - 589824.0 * x6 + 408192.0 * x4 - 77760.0 * x2 + 10395.0)

    def objective(coeffs):
        """
        Compute the upper bound on C4 for the given Hermite coefficients.
        Vectorized on precomputed grid for speed.
        """
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)

        # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
        poly_g = a0 * H0_g + a4 * H4_g + a8 * H8_g + a12 * H12_g
        g_vals = poly_g * gauss_half

        # F(g)(x) = (a0*H0 + a4*H4 + a8*H8 - a12*H12) * exp(-x^2/2)
        # (i^4=1, i^8=1, i^12=-1 for physicists' Hermite)
        poly_fg = a0 * H0_g + a4 * H4_g + a8 * H8_g - a12 * H12_g
        fg_vals = poly_fg * gauss_half

        # L2 norms
        g_sq = g_vals * g_vals
        fg_sq = fg_vals * fg_vals
        norm2_g_sq = np.sum(g_sq) * dx
        norm2_fg_sq = np.sum(fg_sq) * dx

        if norm2_g_sq < 1e-15 or norm2_fg_sq < 1e-15:
            return 1e10

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_fg = np.sqrt(norm2_fg_sq)

        # L4 norms
        g_abs = np.abs(g_vals)
        fg_abs = np.abs(fg_vals)
        g_4 = g_abs * g_abs * g_abs * g_abs
        fg_4 = fg_abs * fg_abs * fg_abs * fg_abs
        norm4_g_4 = np.sum(g_4) * dx
        norm4_fg_4 = np.sum(fg_4) * dx

        if norm4_g_4 < 1e-15 or norm4_fg_4 < 1e-15:
            return 1e10

        norm4_g = norm4_g_4 ** 0.25
        norm4_fg = norm4_fg_4 ** 0.25

        bound = (norm4_g * norm4_fg) / (norm2_g * norm2_fg)
        return bound

    # Evaluate seed
    best_coeffs = seed.copy()
    best_val = objective(seed)

    # Multi-start Nelder-Mead with diverse initial guesses
    rng = np.random.RandomState(42)
    starts = [seed.copy()]

    # Random starts in shrinking neighborhoods
    for i in range(40):
        scale = 0.4 / (1.0 + i * 0.2)
        starts.append(seed + rng.normal(0, scale, size=3))

    # Structured grid of starts
    for a0_t in np.linspace(-1.6, -0.6, 5):
        for a4_t in np.linspace(-0.02, 0.08, 5):
            for a8_t in np.linspace(-0.0005, 0.001, 3):
                starts.append(np.array([a0_t, a4_t, a8_t]))

    n_starts = len(starts)
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 10.0:
            break

        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Final refinement with tighter tolerances
    if time.time() < deadline - 15.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # One more refinement if time allows
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
