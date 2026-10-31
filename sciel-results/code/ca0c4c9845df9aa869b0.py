"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

_START_TIME = time.time()
_TIME_LIMIT = 180.0  # seconds
_SAFETY_MARGIN = 15.0  # seconds to return before hard kill at 210s

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _hermite_polynomials(x):
    """Compute H0, H4, H8, H12 (physicists' Hermite polynomials)."""
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    H12 = (4096*x**12 - 73728*x**10 + 491520*x**8 - 
           1548288*x**6 + 2419200*x**4 - 1548288*x**2 + 1032192)
    return H0, H4, H8, H12

def _compute_objective(coeffs):
    """Compute the upper bound on C4 = A(g)*A(Fourier(g)).
    
    coeffs = [a0, a4, a8]
    a12 is derived from P(0) = 0 constraint.
    """
    a0, a4, a8 = coeffs
    # P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=1032192
    a12 = -(a0 + 12*a4 + 1680*a8) / 1032192.0
    
    # Q(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # R(x) = a0*H0 + a4*H4 + a8*H8 - a12*H12 (Fourier flips sign of H12 term since i^12 = -1)
    
    # Evaluate on a fine grid to estimate sup norms
    x = np.linspace(-8, 8, 20001)
    H0, H4, H8, H12 = _hermite_polynomials(x)
    
    Q = a0*H0 + a4*H4 + a8*H8 + a12*H12
    R = a0*H0 + a4*H4 + a8*H8 - a12*H12
    
    Qg = Q * np.exp(-x**2/2)
    Rg = R * np.exp(-x**2/2)
    
    A_Q = np.max(np.abs(Qg))
    A_R = np.max(np.abs(Rg))
    
    return A_Q * A_R

def _refine_local(x0, max_iters=50):
    """Local refinement using scipy minimize."""
    if _time_left() < 5:
        return x0, _compute_objective(x0)
    
    best_x = x0
    best_f = _compute_objective(x0)
    
    try:
        result = minimize(_compute_objective, x0, method='Nelder-Mead',
                         options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'adaptive': True})
        if result.fun < best_f and np.all(np.isfinite(result.x)):
            best_x = result.x
            best_f = result.fun
    except Exception:
        pass
    
    # Try L-BFGS-B with numerical gradient
    if _time_left() > 10:
        try:
            result2 = minimize(_compute_objective, best_x, method='L-BFGS-B',
                              options={'maxiter': max_iters, 'gtol': 1e-10})
            if result2.fun < best_f and np.all(np.isfinite(result2.x)):
                best_x = result2.x
                best_f = result2.fun
        except Exception:
            pass
    
    return best_x, best_f

def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8).
    
    The degree-12 weight is fixed so g(0) = 0.
    Score is the upper bound on C4 (lower is better).
    """
    # Start with the seed values
    seed = np.array([-113/100, 1/25, 1/3240])
    best_x = seed.copy()
    best_f = _compute_objective(seed)
    
    if _time_left() < 10:
        return best_x
    
    # Phase 1: Global optimization with differential evolution
    if _time_left() > 30:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            de_result = differential_evolution(
                _compute_objective, bounds,
                seed=42, maxiter=100, tol=1e-10,
                popsize=15, mutation=(0.5, 1.5), recombination=0.8,
                polish=True, init='sobol',
                update_interval=1,
                callback=lambda xk, convergence: convergence and _time_left() < 20
            )
            if de_result.fun < best_f and np.all(np.isfinite(de_result.x)):
                best_x = de_result.x
                best_f = de_result.fun
        except Exception:
            pass
    
    if _time_left() < 5:
        return best_x
    
    # Phase 2: Local refinement
    best_x, best_f = _refine_local(best_x, max_iters=100)
    
    if _time_left() < 5:
        return best_x
    
    # Phase 3: Try multiple random restarts with local refinement
    rng = np.random.RandomState(12345)
    for i in range(20):
        if _time_left() < 10:
            break
        # Random perturbation around current best
        scale = 0.5 / (i + 1)
        x_try = best_x + rng.normal(0, scale, size=3)
        x_try, f_try = _refine_local(x_try, max_iters=30)
        if f_try < best_f:
            best_x = x_try
            best_f = f_try
    
    if _time_left() < 5:
        return best_x
    
    # Phase 4: High-precision polish with finer grid
    if _time_left() > 15:
        # Use a denser grid for final evaluation
        x_fine = np.linspace(-10, 10, 50001)
        H0f, H4f, H8f, H12f = _hermite_polynomials(x_fine)
        
        def _objective_fine(coeffs):
            a0, a4, a8 = coeffs
            a12 = -(a0 + 12*a4 + 1680*a8) / 1032192.0
            Q = a0*H0f + a4*H4f + a8*H8f + a12*H12f
            R = a0*H0f + a4*H4f + a8*H8f - a12*H12f
            Qg = Q * np.exp(-x_fine**2/2)
            Rg = R * np.exp(-x_fine**2/2)
            return np.max(np.abs(Qg)) * np.max(np.abs(Rg))
        
        best_f_fine = _objective_fine(best_x)
        
        try:
            result = minimize(_objective_fine, best_x, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14,
                                      'adaptive': True})
            if result.fun < best_f_fine and np.all(np.isfinite(result.x)):
                best_x = result.x
                best_f_fine = result.fun
        except Exception:
            pass
        
        if _time_left() > 10:
            try:
                result2 = minimize(_objective_fine, best_x, method='L-BFGS-B',
                                  options={'maxiter': 200, 'gtol': 1e-12})
                if result2.fun < best_f_fine and np.all(np.isfinite(result2.x)):
                    best_x = result2.x
                    best_f_fine = result2.fun
            except Exception:
                pass
    
    return best_x
# EVOLVE-BLOCK-END
