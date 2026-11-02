import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

# Precompute grid and Hermite polynomials for speed
# The function g(x) decays like x^12 * e^{-x^2/2}.
# The maximum is likely within |x| < 10.
# Let's use a dense grid.
N_GRID = 4000
X_GRID = np.linspace(-12, 12, N_GRID)
Y_GRID = X_GRID ** 2
EXP_GRID = np.exp(-Y_GRID / 2.0)

# Hermite polynomials evaluated on grid
# H0
H0_GRID = np.ones_like(X_GRID)
# H4
H4_GRID = 16 * Y_GRID**2 - 48 * Y_GRID + 12
# H8
Y2 = Y_GRID**2
Y3 = Y2 * Y_GRID
Y4 = Y2 * Y2
H8_GRID = 256 * Y4 - 4480 * Y3 + 26880 * Y2 - 6720 * Y_GRID + 1680
# H12
Y5 = Y4 * Y_GRID
Y6 = Y3 * Y3
H12_GRID = 4096 * Y6 - 245760 * Y5 + 5529600 * Y4 - 45158400 * Y3 + 164430720 * Y2 - 222772500 * Y_GRID + 47900160

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for given coefficients.
    Returns the raw value (lower is better).
    """
    # Calculate c12 to satisfy g(0) = 0
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=47900160
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 47900160.0
    
    # Construct g(x) on the grid
    g_vals = c0 * H0_GRID + c4 * H4_GRID + c8 * H8_GRID + c12 * H12_GRID
    g_vals *= EXP_GRID
    
    # Calculate sup |x * g(x)|
    xg = X_GRID * g_vals
    max_abs = np.max(np.abs(xg))
    
    # Bound is sqrt(2*pi) * (sup |x g(x)|)^2
    return math.sqrt(2.0 * math.pi) * max_abs * max_abs

def _objective(coeffs):
    """
    Objective function for optimization.
    """
    try:
        return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    except Exception:
        return 1e10

def construct():
    """
    Find optimal coefficients by numerical optimization.
    Returns an array of three coefficients [c0, c4, c8].
    """
    start_time = time.time()
    deadline = 175.0  # Safety margin before 210s limit
    
    # Initial guess from current best
    x0 = np.array([-1.13, 0.04, 0.000308642])
    best_val = _objective(x0)
    best_x = x0.copy()
    
    # Phase 1: Global search with Differential Evolution
    if time.time() < deadline - 60:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
            de_result = differential_evolution(
                _objective, bounds,
                maxiter=100, popsize=20,
                tol=1e-8, seed=42,
                polish=True,
                update_interval=10
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with multiple starting points
    # Generate diverse starting points around the best found so far and known good regions
    start_points = [
        best_x.copy(),
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-1.2, 0.03, 0.00025]),
        np.array([-0.9, 0.06, 0.0005]),
        np.array([-1.3, 0.02, 0.0002]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        try:
            # Use Nelder-Mead for local search
            res = minimize(
                _objective, sp, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective, best_x, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
