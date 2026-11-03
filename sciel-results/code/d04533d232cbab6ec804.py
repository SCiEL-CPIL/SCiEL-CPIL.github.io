"""Rewritten program: robust global search for the Hermite-Gaussian coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import differential_evolution, minimize
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

    # High-resolution grid (Hermite-Gaussian decays fast; -15..15 is ample)
    N = 8000
    x = np.linspace(-15.0, 15.0, N)
    dx = x[1] - x[0]

    # Precompute basis on grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)

    # Precompute the L2 norm squared of each basis and cross terms for speed
    # g = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * gauss, c12 = -(c0 + 3 c4 + 105 c8)/9240
    # Let's precompute the "effective" basis functions including c12 contribution.
    # c12 * H12 = -(c0 + 3 c4 + 105 c8)/9240 * H12
    # So g/gauss = c0*(H0 - H12/9240) + c4*(H4 - 3 H12/9240) + c8*(H8 - 105 H12/9240)
    # Define effective basis vectors:
    B0 = H0 - H12 / h12_0
    B4 = H4 - (h4_0 / h12_0) * H12
    B8 = H8 - (h8_0 / h12_0) * H12

    # Precompute B_i * gauss and their L2 norms
    Bg0 = B0 * gauss
    Bg4 = B4 * gauss
    Bg8 = B8 * gauss

    # L2 norm squared matrix (3x3)
    G00 = np.sum(Bg0 * Bg0) * dx
    G44 = np.sum(Bg4 * Bg4) * dx
    G88 = np.sum(Bg8 * Bg8) * dx
    G04 = np.sum(Bg0 * Bg4) * dx
    G08 = np.sum(Bg0 * Bg8) * dx
    G48 = np.sum(Bg4 * Bg8) * dx

    def compute_bound(c0, c4, c8):
        # g = c0*Bg0 + c4*Bg4 + c8*Bg8
        g = c0 * Bg0 + c4 * Bg4 + c8 * Bg8
        # L2 norm squared via precomputed Gram matrix (faster & exact for this combo)
        norm_sq = (c0*c0*G00 + c4*c4*G44 + c8*c8*G88
                   + 2.0*c0*c4*G04 + 2.0*c0*c8*G08 + 2.0*c4*c8*G48)
        if norm_sq < 1e-18:
            return 1e10
        inv_sqrt = 1.0 / np.sqrt(norm_sq)
        g_norm = g * inv_sqrt
        l1_g = np.sum(np.abs(g_norm)) * dx
        # Fourier transform of these even Hermite-Gaussians has same L1 norm (symmetry)
        return l1_g * l1_g

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    seed = [c0_init, c4_init, c8_init]
    best_score = compute_bound(*seed)
    best_coeffs = np.array(seed)

    def time_ok(margin=20.0):
        return (time.time() - start_time) < (deadline - margin)

    # ---- Phase 1: Differential Evolution (global, robust) ----
    if time_ok(30.0):
        # Search box: allow generous range around seed
        bounds = [
            (c0_init - 3.0, c0_init + 3.0),
            (c4_init - 0.5, c4_init + 0.5),
            (c8_init - 0.05, c8_init + 0.05),
        ]
        try:
            de_res = differential_evolution(
                lambda c: compute_bound(*c),
                bounds,
                seed=12345,
                maxiter=400,
                popsize=30,
                tol=1e-10,
                mutation=(0.3, 1.0),
                recombination=0.9,
                init='latinhypercube',
                polish=False,
                updating='deferred',
                callback=lambda p, c: (time.time() - start_time) < deadline,
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_coeffs = np.array(de_res.x)
        except Exception:
            pass

    # ---- Phase 2: Multi-start Nelder-Mead local polish from several points ----
    if time_ok(15.0):
        rng = np.random.default_rng(7)
        # Candidate starting points: best so far + seed + random perturbations
        starts = [best_coeffs.tolist(), seed]
        for _ in range(12):
            pert = [
                c0_init + rng.normal(0, 1.0),
                c4_init + rng.normal(0, 0.2),
                c8_init + rng.normal(0, 0.02),
            ]
            starts.append(pert)
        for s in starts:
            if not time_ok(10.0):
                break
            try:
                res = minimize(
                    lambda c: compute_bound(*c),
                    s,
                    method='Nelder-Mead',
                    options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-13,
                             'adaptive': True, 'maxfev': 2000},
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = np.array(res.x)
            except Exception:
                pass

    # ---- Phase 3: Tight final polish ----
    if time_ok(5.0):
        try:
            res = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16,
                         'adaptive': True, 'maxfev': 10000},
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.array(res.x)
        except Exception:
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
