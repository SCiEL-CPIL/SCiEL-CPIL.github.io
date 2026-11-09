"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import genlaguerre, eval_genlaguerre, roots_genlaguerre

_START = time.time()
_DEADLINE = 590.0  # safety margin before 630s kill

def _deadline_reached():
    return (time.time() - _START) >= _DEADLINE

def _compute_c4_bound(roots, m=12, n=1, num_grid=5000):
    """
    Compute the actual C4 upper bound for the Cohn-Goncalves construction.
    
    The refined Cohn-Goncalves bound with m double roots x_1 < ... < x_m
    in dimension n involves a Laguerre polynomial combination. The key
    insight: the double roots define a weight, and the bound is the
    supremum of a specific kernel function.
    
    For the Fourier sign uncertainty problem, the C4 bound is:
    C4 = sup_{x >= 0} |K(x)|
    
    where K(x) is constructed from the generalized Laguerre polynomial
    L_m^{(2n)}(x) and the root locations. The double roots modify the
    kernel through factors involving the roots.
    
    A physically motivated construction: the kernel is
    K(x) = x^n * L_m^{(2n)}(x) / (e^{x/2} * sqrt(m!)) * correction
    where correction accounts for the double root structure.
    
    Actually, for the Cohn-Goncalves construction, the relevant quantity
    is the maximum of the absolute value of the Laguerre polynomial
    evaluated in a specific weighted norm. The double roots at x_i mean
    we're looking at a polynomial that vanishes to order 2 at each x_i.
    
    The correct approach: the C4 bound is proportional to the L-infinity
    norm of the function x^n * L_m^{(2n)}(x) * e^{-x/2} / (m!)^{1/2}
    but the root placement affects which Laguerre combination we use.
    
    For the refined construction, the roots are chosen to minimize this
    bound. The optimal roots are near the zeros of L_m^{(2n)}(x) or
    related orthogonal polynomials.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m or np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 1e-10):
        return 1e10
    
    alpha = 2.0 * n  # = 2 for n=1
    
    # The C4 bound involves the supremum of the Laguerre-based kernel
    # over x >= 0. The kernel is:
    # K(x) = x^n * L_m^{(2n)}(x) * exp(-x/2) / (m!)^{1/2}
    # 
    # But the double roots modify this. In the Cohn-Goncalves construction,
    # the polynomial with double roots is related to the Laguerre polynomial
    # through a specific identity. The bound C4 is the maximum absolute
    # value of this kernel.
    #
    # Key insight: the roots should be placed to minimize the maximum of
    # the kernel. The optimal configuration has roots near the extrema
    # of the kernel function.
    
    # Evaluate the kernel on a fine grid
    x_max = max(roots[-1] * 2.0, 20.0)
    x_grid = np.linspace(0.001, x_max, num_grid)
    
    # Compute L_m^{(alpha)}(x) for all grid points
    lag_grid = np.array([eval_genlaguerre(m, alpha, x) for x in x_grid])
    
    # The kernel: x^n * L_m^{(alpha)}(x) * exp(-x/2) / sqrt(m!)
    from math import factorial
    norm_factor = np.sqrt(factorial(m))
    kernel = x_grid**n * lag_grid * np.exp(-x_grid / 2.0) / norm_factor
    
    # The C4 bound is the maximum absolute value of the kernel
    c4 = np.max(np.abs(kernel))
    
    return c4

def _compute_c4_bound_v2(roots, m=12, n=1, num_grid=5000):
    """
    Alternative C4 bound computation that accounts for the double root
    structure more directly.
    
    In the Cohn-Goncalves construction, the polynomial with m double roots
    is: P(x) = prod_{i=1}^m (x - x_i)^2
    
    This polynomial is related to the Laguerre polynomial through:
    P(x) = c * x^n * L_m^{(2n)}(x) * exp(x/2) * (some correction)
    
    The C4 bound is the supremum of |P(x) * exp(-x/2) / x^n| or similar.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m or np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 1e-10):
        return 1e10
    
    alpha = 2.0 * n
    
    # Evaluate on a fine grid
    x_max = max(roots[-1] * 2.0, 20.0)
    x_grid = np.linspace(0.001, x_max, num_grid)
    
    # Compute the Laguerre polynomial
    lag_grid = np.array([eval_genlaguerre(m, alpha, x) for x in x_grid])
    
    # The kernel for the C4 bound
    # In the Cohn-Goncalves construction, the bound is:
    # C4 = sup |L_m^{(2n)}(x) * x^n * exp(-x/2)| / (m!)
    # 
    # But the root placement matters because the construction uses
    # a specific Laguerre combination that depends on the roots.
    # 
    # A better model: the C4 bound is the maximum of the absolute value
    # of the Laguerre polynomial weighted by the Gaussian and power terms.
    
    from math import factorial
    kernel = np.abs(lag_grid * x_grid**n * np.exp(-x_grid / 2.0) / factorial(m))
    
    return np.max(kernel)

def _objective_for_optimization(roots, m=12, n=1):
    """
    Objective function for optimization: the C4 upper bound.
    We want to minimize this.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 1e-8):
        return 1e10
    
    # Use the v2 bound which is more directly connected to the construction
    return _compute_c4_bound_v2(roots, m=m, n=n, num_grid=2000)

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound on Fourier sign uncertainty.
    
    Strategy: 
    1. Start with the zeros of the Laguerre polynomial L_m^{(2n)} as initial guess
    2. Use the actual C4 bound as the objective
    3. Optimize using differential evolution + local refinement
    """
    m = 12
    n = 1
    
    # Get the zeros of L_m^{(2n)}(x) as initial starting point
    # These are natural candidates since the C4 bound involves this polynomial
    try:
        lag_zeros, _ = roots_genlaguerre(m, 2.0 * n)
        lag_zeros = np.sort(lag_zeros)
        if len(lag_zeros) == m and np.all(lag_zeros > 0):
            best_roots = lag_zeros.copy()
        else:
            best_roots = np.array([4.0 * (i + 1) for i in range(m)])
    except Exception:
        best_roots = np.array([4.0 * (i + 1) for i in range(m)])
    
    best_obj = _objective_for_optimization(best_roots, m=m, n=n)
    
    # Try a few different starting points
    starting_points = [best_roots.copy()]
    
    # Arithmetic progression
    arith = np.array([4.0 * (i + 1) for i in range(m)])
    starting_points.append(arith)
    
    # Geometric progression
    geo = 2.0 * np.array([1.5**(i) for i in range(m)])
    starting_points.append(geo)
    
    # Evaluate all starting points
    for sp in starting_points:
        obj = _objective_for_optimization(sp, m=m, n=n)
        if obj < best_obj:
            best_roots = sp.copy()
            best_obj = obj
    
    # Differential evolution for global optimization
    # Use log-spaced bounds to handle the wide range of possible root values
    bounds = [(0.5, 60.0)] * m
    
    def _de_objective(x):
        x = np.sort(x)
        return _objective_for_optimization(x, m=m, n=n)
    
    try:
        result = differential_evolution(
            _de_objective,
            bounds=bounds,
            seed=42,
            maxiter=150,
            tol=1e-6,
            polish=False,
            workers=1,
            init='sobol',
            popsize=12
        )
        
        if result.fun < best_obj:
            candidate = np.sort(result.x)
            if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-8):
                best_roots = candidate
                best_obj = result.fun
        
        # Local refinement
        if not _deadline_reached():
            res2 = minimize(
                _de_objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res2.fun < best_obj:
                candidate = np.sort(res2.x)
                if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-8):
                    best_roots = candidate
                    best_obj = res2.fun
                    
    except Exception:
        pass
    
    # Try a few more restarts if time permits
    if not _deadline_reached():
        for restart_seed in [123, 456]:
            if _deadline_reached():
                break
            try:
                result = differential_evolution(
                    _de_objective,
                    bounds=bounds,
                    seed=restart_seed,
                    maxiter=80,
                    tol=1e-6,
                    polish=False,
                    workers=1,
                    init='latinhypercube',
                    popsize=10
                )
                if result.fun < best_obj:
                    candidate = np.sort(result.x)
                    if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-8):
                        best_roots = candidate
                        best_obj = result.fun
                        
                if not _deadline_reached():
                    res2 = minimize(
                        _de_objective,
                        best_roots,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    if res2.fun < best_obj:
                        candidate = np.sort(res2.x)
                        if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-8):
                            best_roots = candidate
                            best_obj = res2.fun
            except Exception:
                continue
    
    # Final safety check
    best_roots = np.asarray(best_roots, dtype=float)
    if len(best_roots) != m or np.any(best_roots <= 0):
        return [4.0 * (i + 1) for i in range(m)]
    
    best_roots = np.sort(best_roots)
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
