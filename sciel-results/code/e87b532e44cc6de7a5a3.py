"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.special import factorial

# Deadline guard: 210s limit, use 195s for safety
_DEADLINE = time.time() + 195

def _time_remaining():
    return _DEADLINE - time.time()

def _compute_objective(c0, c4, c8):
    """
    Compute the upper bound: (||g||_1^2) / (||g||_2^2)
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    Constraint: g(0) = 0 => c12 is determined.
    """
    # H_n(0) values
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Exact L2 norm squared
    # ||g||_2^2 = sqrt(pi) * sum c_n^2 * 2^n * n!
    # 2^0 * 0! = 1
    # 2^4 * 4! = 16 * 24 = 384
    # 2^8 * 8! = 256 * 40320 = 10321920
    # 2^12 * 12! = 4096 * 479001600 = 1962010560000
    norm2_sq = np.sqrt(np.pi) * (
        c0**2 * 1.0 +
        c4**2 * 384.0 +
        c8**2 * 10321920.0 +
        c12**2 * 1962010560000.0
    )
    
    if norm2_sq <= 0 or not np.isfinite(norm2_sq):
        return 1e10
    
    # Numerical L1 norm
    # Define g(x)
    def g_abs(x):
        # Hermite polynomials
        # H0 = 1
        # H4 = 16x^4 - 48x^2 + 12
        # H8 = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
        # H12 = 4096x^12 - 147456x^10 + 2090112x^8 - 12902400x^6 + 34406400x^4 - 34406400x^2 + 10321920
        
        # Use vectorized operations for speed if x is array, but quad passes scalar
        # We can precompute coefficients for efficiency if needed, but quad is adaptive
        h0 = 1.0
        h4 = 16.0*x**4 - 48.0*x**2 + 12.0
        h8 = 256.0*x**8 - 4608.0*x**6 + 23040.0*x**4 - 34560.0*x**2 + 10080.0
        h12 = (4096.0*x**12 - 147456.0*x**10 + 2090112.0*x**8 - 
               12902400.0*x**6 + 34406400.0*x**4 - 34406400.0*x**2 + 10321920.0)
        
        poly = c0*h0 + c4*h4 + c8*h8 + c12*h12
        val = poly * np.exp(-x**2 / 2.0)
        return abs(val)
    
    try:
        # Integrate over a large enough range. Gaussian decays fast.
        # quad with inf bounds is usually safe for Gaussians
        norm1, err = quad(g_abs, -np.inf, np.inf, limit=100)
    except Exception:
        # Fallback to numerical integration on a grid
        x_grid = np.linspace(-15, 15, 10000)
        # Recalculate for grid
        h0 = np.ones_like(x_grid)
        h4 = 16.0*x_grid**4 - 48.0*x_grid**2 + 12.0
        h8 = 256.0*x_grid**8 - 4608.0*x_grid**6 + 23040.0*x_grid**4 - 34560.0*x_grid**2 + 10080.0
        h12 = (4096.0*x_grid**12 - 147456.0*x_grid**10 + 2090112.0*x_grid**8 - 
               12902400.0*x_grid**6 + 34406400.0*x_grid**4 - 34406400.0*x_grid**2 + 10321920.0)
        poly = c0*h0 + c4*h4 + c8*h8 + c12*h12
        vals = np.abs(poly * np.exp(-x_grid**2 / 2.0))
        norm1 = np.trapz(vals, x_grid)
        
    if not np.isfinite(norm1) or norm1 < 0:
        return 1e10
        
    bound = (norm1**2) / norm2_sq
    return bound

def construct():
    """
    Find coefficients c0, c4, c8 that minimize the uncertainty bound.
    Returns np.array([c0, c4, c8]).
    """
    # Seed coefficients from the prompt
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = seed_result.copy()
    best_score = _compute_objective(*seed_result)
    
    # 1. Local optimization from seed
    if _time_remaining() > 10:
        try:
            res = minimize(_compute_objective, seed_result, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
            
    # 2. Global search around seed with Differential Evolution
    # Use tight bounds to avoid exploring irrelevant regions
    if _time_remaining() > 30:
        try:
            # Seed is approx [-1.13, 0.04, 0.0003]
            # Allow +/- 50% variation
            bounds = [
                (seed_result[0] - 0.5, seed_result[0] + 0.5),
                (seed_result[1] - 0.02, seed_result[1] + 0.02),
                (seed_result[2] - 0.0002, seed_result[2] + 0.0002)
            ]
            
            de_result = differential_evolution(
                _compute_objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                polish=True
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
            
    return best_result

# EVOLVE-BLOCK-END
