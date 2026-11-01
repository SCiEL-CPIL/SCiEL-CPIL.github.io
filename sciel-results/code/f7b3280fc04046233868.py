"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math
from scipy.special import eval_hermite

# Precompute Hermite values at 0
H0_0 = float(eval_hermite(0, 0))
H4_0 = float(eval_hermite(4, 0))
H8_0 = float(eval_hermite(8, 0))
H12_0 = float(eval_hermite(12, 0))

# Normalization: ||H_k e^{-x^2/2}||^2 = 2^k k! sqrt(pi)
N0 = 1.0
N4 = 2.0**4 * math.factorial(4)
N8 = 2.0**8 * math.factorial(8)
N12 = 2.0**12 * math.factorial(12)

inv_sqrt_N0 = 1.0 / math.sqrt(N0)
inv_sqrt_N4 = 1.0 / math.sqrt(N4)
inv_sqrt_N8 = 1.0 / math.sqrt(N8)
inv_sqrt_N12 = 1.0 / math.sqrt(N12)

# Precompute a grid for numerical verification of the bound
# Use a dense grid to compute actual L2 norms if needed
_x_grid = np.linspace(-10, 10, 200001)
_dx = _x_grid[1] - _x_grid[0]
_exp_grid = np.exp(-_x_grid**2 / 2.0)

# Precompute H_k on the grid for k=0,4,8,12
_H0_grid = np.ones_like(_x_grid)
_H4_grid = eval_hermite(4, _x_grid)
_H8_grid = eval_hermite(8, _x_grid)
_H12_grid = eval_hermite(12, _x_grid)


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 = A(f) * A(Fourier(f)).
    Uses the analytical diagonal formula for <x^2>.
    """
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    c0 = a0 * inv_sqrt_N0
    c4 = a4 * inv_sqrt_N4
    c8 = a8 * inv_sqrt_N8
    c12 = a12 * inv_sqrt_N12
    
    c0_sq = c0 * c0
    c4_sq = c4 * c4
    c8_sq = c8 * c8
    c12_sq = c12 * c12
    
    total_norm = c0_sq + c4_sq + c8_sq + c12_sq
    
    if total_norm < 1e-15:
        return float('inf')
    
    x2 = (0.5 * c0_sq + 4.5 * c4_sq + 8.5 * c8_sq + 12.5 * c12_sq) / total_norm
    
    return x2


def _compute_bound_numerical(a0, a4, a8):
    """
    Numerically compute A(f)*A(F) using grid integration.
    This serves as a verification and potentially more accurate method.
    """
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    g = (a0 * _H0_grid + a4 * _H4_grid + a8 * _H8_grid + a12 * _H12_grid) * _exp_grid
    
    # ||g||^2
    norm_sq = np.sum(g**2) * _dx
    if norm_sq < 1e-15:
        return float('inf')
    
    # ||x*g||^2
    xg_sq = np.sum((_x_grid * g)**2) * _dx
    
    # A(f) = sqrt(||x*g||^2 / ||g||^2)
    A_f = math.sqrt(xg_sq / norm_sq)
    
    # Fourier transform: for Hermite-Gaussians, F(H_k e^{-x^2/2}) = (-i)^k H_k e^{-xi^2/2}
    # For even k, (-i)^k = 1, so F(g)(xi) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)(xi) * exp(-xi^2/2)
    # Same form as g, so A(F) = A(f)
    A_F = A_f
    
    return A_f * A_F


def _compute_gradient(a0, a4, a8):
    """Analytic gradient of the analytical bound."""
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    c0 = a0 * inv_sqrt_N0
    c4 = a4 * inv_sqrt_N4
    c8 = a8 * inv_sqrt_N8
    c12 = a12 * inv_sqrt_N12
    
    c0_sq = c0 * c0
    c4_sq = c4 * c4
    c8_sq = c8 * c8
    c12_sq = c12 * c12
    
    total_norm = c0_sq + c4_sq + c8_sq + c12_sq
    
    if total_norm < 1e-15:
        return np.array([0.0, 0.0, 0.0])
    
    V = 0.5 * c0_sq + 4.5 * c4_sq + 8.5 * c8_sq + 12.5 * c12_sq
    N = total_norm
    
    dc0_da0 = inv_sqrt_N0
    dc4_da4 = inv_sqrt_N4
    dc8_da8 = inv_sqrt_N8
    dc12_da0 = inv_sqrt_N12 * (-H0_0 / H12_0)
    dc12_da4 = inv_sqrt_N12 * (-H4_0 / H12_0)
    dc12_da8 = inv_sqrt_N12 * (-H8_0 / H12_0)
    
    dV_da0 = c0 * dc0_da0 + 25.0 * c12 * dc12_da0
    dN_da0 = 2.0 * c0 * dc0_da0 + 2.0 * c12 * dc12_da0
    
    dV_da4 = 9.0 * c4 * dc4_da4 + 25.0 * c12 * dc12_da4
    dN_da4 = 2.0 * c4 * dc4_da4 + 2.0 * c12 * dc12_da4
    
    dV_da8 = 17.0 * c8 * dc8_da8 + 25.0 * c12 * dc12_da8
    dN_da8 = 2.0 * c8 * dc8_da8 + 2.0 * c12 * dc12_da8
    
    grad_a0 = (dV_da0 * N - V * dN_da0) / (N * N)
    grad_a4 = (dV_da4 * N - V * dN_da4) / (N * N)
    grad_a8 = (dV_da8 * N - V * dN_da8) / (N * N)
    
    return np.array([grad_a0, grad_a4, grad_a8])


def construct():
    """
    Returns three coefficients (a0, a4, a8) for the Hermite combination.
    Uses differential evolution for global search, then L-BFGS-B for local polish.
    """
    start_time = time.time()
    deadline = start_time + 180.0 - 15.0  # 15 second safety margin
    
    # Seed value for reference
    seed_val = _compute_bound(-113.0/100.0, 1.0/25.0, 1.0/3240.0)
    
    # Phase 1: Global search with differential evolution
    # Use a bounded box around reasonable values
    bounds = [
        (-5.0, 5.0),    # a0
        (-1.0, 1.0),    # a4
        (-0.01, 0.01),  # a8
    ]
    
    best_result = None
    best_obj = float('inf')
    
    # Run differential evolution with limited budget
    try:
        def de_objective(x):
            if time.time() > deadline - 30:
                return 1e10
            return _compute_bound(x[0], x[1], x[2])
        
        de_result = differential_evolution(
            de_objective, bounds,
            maxiter=50, popsize=20,
            tol=1e-10, mutation=(0.3, 1.0), recombination=0.7,
            seed=42, polish=False, init='sobol'
        )
        
        if de_result.fun < best_obj:
            best_obj = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    # Phase 2: Local polish with L-BFGS-B from multiple starting points
    start_points = []
    
    # Include the seed
    start_points.append(np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0]))
    
    # Include DE result if valid
    if best_result is not None:
        start_points.append(best_result)
    
    # Add diverse starting points
    rng = np.random.RandomState(42)
    for _ in range(20):
        sp = np.array([
            rng.uniform(-3.0, 3.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.005, 0.005),
        ])
        start_points.append(sp)
    
    def objective(coeffs):
        if time.time() > deadline:
            return 1e10
        return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    
    def objective_grad(coeffs):
        if time.time() > deadline:
            return np.array([0.0, 0.0, 0.0])
        return _compute_gradient(coeffs[0], coeffs[1], coeffs[2])
    
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(objective, sp, jac=objective_grad, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_result is not None and time.time() < deadline - 5:
        try:
            result = minimize(objective, best_result, jac=objective_grad, method='L-BFGS-B',
                            options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-18})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is None:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    return best_result
# EVOLVE-BLOCK-END
