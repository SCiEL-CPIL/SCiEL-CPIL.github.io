"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
from scipy.integrate import quad
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

    # Precompute grid for fast approximate objective during optimization
    x_grid = np.linspace(-15, 15, 2000)
    dx = x_grid[1] - x_grid[0]
    
    H0_grid = eval_hermite(0, x_grid)
    H4_grid = eval_hermite(4, x_grid)
    H8_grid = eval_hermite(8, x_grid)
    H12_grid = eval_hermite(12, x_grid)
    gauss_grid = np.exp(-x_grid**2 / 2.0)

    def g_val(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        # Evaluate Hermite polynomials at x
        H0 = eval_hermite(0, x)
        H4 = eval_hermite(4, x)
        H8 = eval_hermite(8, x)
        H12 = eval_hermite(12, x)
        poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        return poly * np.exp(-x**2 / 2.0)

    def compute_bound_fast(c0, c4, c8):
        """Fast approximation using fixed grid for optimization."""
        if time.time() > deadline:
            return 1e10
        c12 = get_c12(c0, c4, c8)
        poly = c0 * H0_grid + c4 * H4_grid + c8 * H8_grid + c12 * H12_grid
        g = poly * gauss_grid
        
        norm_sq = np.sum(g**2) * dx
        if norm_sq < 1e-15:
            return 1e10
        
        g_norm = g / np.sqrt(norm_sq)
        l1_g = np.sum(np.abs(g_norm)) * dx
        return l1_g * l1_g

    def compute_bound_exact(c0, c4, c8):
        """Exact calculation using adaptive quadrature for final evaluation."""
        if time.time() > deadline:
            return 1e10
        
        def g_sq(x):
            val = g_val(x, c0, c4, c8)
            return val * val
        
        def abs_g(x):
            val = g_val(x, c0, c4, c8)
            return np.abs(val)

        # Integrate over real line. Gaussian decays fast, [-20, 20] is sufficient.
        norm_sq, _ = quad(g_sq, -20, 20, limit=100, epsabs=1e-10, epsrel=1e-10)
        if norm_sq < 1e-15:
            return 1e10
        
        l1_g, _ = quad(abs_g, -20, 20, limit=100, epsabs=1e-10, epsrel=1e-10)
        
        # Normalize
        l1_g_norm = l1_g / np.sqrt(norm_sq)
        
        return l1_g_norm * l1_g_norm

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0

    best_score_fast = compute_bound_fast(c0_init, c4_init, c8_init)
    best_coeffs = [c0_init, c4_init, c8_init]

    # Strategy 1: Differential Evolution for global search
    # This is robust for non-convex landscapes and handles the 3D space well.
    # We use the fast grid-based objective for speed.
    try:
        bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
        # Differential evolution can be slow, so we limit population and maxiter
        # but it explores globally.
        res_de = differential_evolution(
            compute_bound_fast,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            popsize=15,
            polish=False  # We will polish separately
        )
        if res_de.fun < best_score_fast:
            best_score_fast = res_de.fun
            best_coeffs = res_de.x.tolist()
    except Exception:
        pass

    # Strategy 2: Multi-start L-BFGS-B for local refinement
    # Use the result from DE as one start, plus random starts around seed.
    np.random.seed(42)
    n_starts = 20
    max_iter_per_start = 300

    # Add DE result to start list
    start_points = [best_coeffs[:]]
    
    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate initial guess around seed
        if i == 0:
            x0 = [c0_init, c4_init, c8_init]
        else:
            scale_c0 = 0.5 * (1 + i * 0.1)
            scale_c4 = 0.1 * (1 + i * 0.1)
            scale_c8 = 0.01 * (1 + i * 0.1)
            x0 = [
                c0_init + np.random.randn() * scale_c0,
                c4_init + np.random.randn() * scale_c4,
                c8_init + np.random.randn() * scale_c8
            ]
        start_points.append(x0)

    for x0 in start_points:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(
                lambda c: compute_bound_fast(*c),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_start, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_score_fast:
                best_score_fast = res.fun
                best_coeffs = res.x.tolist()
        except Exception:
            pass

    # Final Polish: Use exact quadrature to verify and fine-tune
    # Since the grid approximation might have slight bias, we do a final
    # high-precision check and potentially a tiny local adjustment using exact objective.
    
    # First, get the exact score of the current best
    current_exact = compute_bound_exact(*best_coeffs)
    
    # Perform a very short, tight L-BFGS-B run using the EXACT objective
    # This ensures the final coefficients are optimal for the true integral, not the grid approx.
    if time.time() < deadline - 5.0:
        try:
            res_exact = minimize(
                lambda c: compute_bound_exact(*c),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_exact.fun < current_exact:
                current_exact = res_exact.fun
                best_coeffs = res_exact.x.tolist()
        except Exception:
            pass

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
