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

    # Precompute grid - use a finer grid for better accuracy
    x = np.linspace(-20, 20, 50000)
    
    # Precompute Hermite polynomials on grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)
    
    # Precompute weights for trapezoidal rule
    dx = x[1] - x[0]
    weights = np.ones_like(x)
    weights[0] = 0.5
    weights[-1] = 0.5

    def compute_bound(c0, c4, c8):
        if time.time() > deadline:
            return 1e10
        c12 = get_c12(c0, c4, c8)

        # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
        poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        g = poly * gauss

        # L2 norm squared using weighted sum for speed and accuracy
        norm_sq = np.sum(weights * g**2) * dx
        if norm_sq < 1e-15:
            return 1e10

        # Normalize to L2 = 1
        g_norm = g / np.sqrt(norm_sq)

        # L1 norm of g using weighted sum
        l1_g = np.sum(weights * np.abs(g_norm)) * dx

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

    # Strategy:
    # 1. Global exploration with Differential Evolution (robust to non-convexity)
    # 2. Local refinement with L-BFGS-B (fast convergence)
    # 3. Final polish with Nelder-Mead (robust for final convergence)

    # Phase 1: Differential Evolution for global search
    # Bounds for coefficients: based on seed magnitude and reasonable exploration
    bounds = [
        (-5.0, 5.0),       # c0
        (-0.5, 0.5),       # c4
        (-0.01, 0.01)      # c8
    ]

    if time.time() < deadline - 40.0:
        try:
            # Use DE with moderate population size and iterations
            res_de = differential_evolution(
                lambda c: compute_bound(*c),
                bounds,
                maxiter=80,
                popsize=12,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42,
                polish=False
            )
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_coeffs = res_de.x.tolist()
        except Exception:
            pass

    # Phase 2: Local refinement with L-BFGS-B from multiple starts
    starts = [best_coeffs, [c0_init, c4_init, c8_init]]
    
    # Add random perturbations around the best point
    np.random.seed(42)
    for i in range(8):
        if time.time() > deadline - 25.0:
            break
        scale = 0.05
        x0 = [
            best_coeffs[0] + np.random.randn() * scale,
            best_coeffs[1] + np.random.randn() * scale * 0.1,
            best_coeffs[2] + np.random.randn() * scale * 0.01
        ]
        starts.append(x0)

    for x0 in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            res_lbfgs = minimize(
                lambda c: compute_bound(*c),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res_lbfgs.fun < best_score:
                best_score = res_lbfgs.fun
                best_coeffs = res_lbfgs.x.tolist()
        except Exception:
            pass

    # Phase 3: Final polish with Nelder-Mead for robust convergence
    if time.time() < deadline - 5.0 and best_score < 1e10:
        try:
            res_nm = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_coeffs = res_nm.x.tolist()
        except Exception:
            pass

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
