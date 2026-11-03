"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre

_start_time = time.time()
_DEADLINE = 600.0  # seconds, with safety margin

def _c4_bound(roots):
    """
    Compute the C4 upper bound given 12 positive double-root locations.
    For the Cohn-Goncalves Laguerre construction with m=12, n=1,
    the C4 bound is derived from the Laguerre polynomial combination.
    
    The construction uses P(x) = prod_i (x - r_i)^2, and the C4 bound
    involves the ratio of moments under the Laguerre weight e^{-x}.
    """
    roots = np.asarray(roots, dtype=float)
    # Ensure positive
    roots = np.maximum(roots, 1e-10)
    
    # Build the polynomial coefficients for P(x) = prod (x - r_i)^2
    # Use log-space for numerical stability
    # P(x) = prod_i (x - r_i)^2
    
    # Compute moments: M_k = int_0^inf e^{-x} x^k P(x) dx
    # We need the C4 bound which typically involves:
    # C4 = [int e^{-x} P(x)^2 dx] / [int e^{-x} P(x) dx]^2 * some normalization
    
    # Actually for the sign uncertainty, the C4 bound is:
    # C4 = sup |f|^4 / |f|^2 type quantity, bounded via Laguerre expansion
    
    # A common formulation: if f(x) = e^{-x/2} P(x) where P is the Laguerre combination,
    # then the uncertainty involves ratios of integrals.
    
    # Let's use the direct moment approach:
    # P(x) = prod_i (x - r_i)^2
    # Compute int_0^inf e^{-x} P(x) dx and int_0^inf e^{-x} P(x)^2 dx
    
    # P(x) has degree 24. P(x)^2 has degree 48.
    # We can compute these via the Laguerre polynomial expansion.
    
    # Alternative: use the fact that int_0^inf e^{-x} x^k dx = k!
    # So we need the coefficients of P(x) and P(x)^2.
    
    # Build P(x) coefficients
    # Start with 1, multiply by (x - r_i)^2 for each i
    coeffs = np.array([1.0])
    for r in roots:
        # (x - r)^2 = x^2 - 2r x + r^2
        factor = np.array([1.0, -2.0*r, r*r])
        coeffs = np.convolve(coeffs, factor)
    
    # coeffs[k] is coefficient of x^k, so P(x) = sum coeffs[k] x^k
    # int_0^inf e^{-x} x^k dx = k!
    # int_0^inf e^{-x} P(x) dx = sum_k coeffs[k] * k!
    
    n_max = len(coeffs) - 1
    factorials = np.array([1.0])
    for i in range(1, n_max + 1):
        factorials.append(factorials[-1] * i)
    
    moment_P = np.sum(coeffs * factorials)
    
    # P(x)^2 = coeffs * coeffs (convolution)
    coeffs_sq = np.convolve(coeffs, coeffs)
    n_max_sq = len(coeffs_sq) - 1
    factorials_sq = np.array([1.0])
    for i in range(1, n_max_sq + 1):
        factorials_sq.append(factorials_sq[-1] * i)
    
    moment_P2 = np.sum(coeffs_sq * factorials_sq)
    
    # C4 bound: the standard form is moment_P2 / moment_P^2
    # But we need to be careful about the exact definition.
    # For the Cohn-Goncalves construction, C4 = moment_P2 / moment_P^2
    # This is the variance-like quantity that gives the upper bound.
    
    if moment_P == 0:
        return 1e10
    
    c4 = moment_P2 / (moment_P ** 2)
    return c4

def _validate_roots(roots):
    """Check that roots are positive and distinct."""
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return False
    if np.any(roots <= 0):
        return False
    # Check distinctness
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-8):
        return False
    return True

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    """
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed - 30  # 30s safety margin
    
    if remaining < 10:
        # Not enough time, return a reasonable default
        return [4.0 * (i + 1) for i in range(12)]
    
    # Initial guess: use the known Laguerre zeros or a structured guess
    # The zeros of L_12^{(0)}(x) are known and can be computed
    # Let's start with those as initial guess
    from scipy.special import roots_genlaguerre
    
    # Roots of L_12^{(0)}(x)
    laguerre_roots = roots_genlaguerre(12, 0)[0]
    # These are 12 positive roots
    x0 = laguerre_roots.copy()
    
    # Try differential evolution for global optimization
    best_result = None
    best_c4 = _c4_bound(x0)
    
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed - 30
    
    if remaining > 60:
        # Use differential evolution for global search
        def objective(x):
            elapsed_inner = time.time() - _start_time
            if elapsed_inner > _DEADLINE - 30:
                return 1e10
            return _c4_bound(x)
        
        # Bounds for each root: reasonable range
        bounds = [(0.1, 100.0)] * 12
        
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                polish=True,
                popsize=15,
                time_limit=min(remaining - 30, 500)
            )
            if result.success or result.fun < best_c4:
                best_c4 = result.fun
                best_result = result.x
        except Exception as e:
            pass
    
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed - 30
    
    if remaining > 30:
        # Refine with L-BFGS-B
        def objective_refine(x):
            elapsed_inner = time.time() - _start_time
            if elapsed_inner > _DEADLINE - 30:
                return 1e10
            return _c4_bound(x)
        
        def jac_refine(x):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(12):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (objective_refine(x_plus) - objective_refine(x_minus)) / (2 * eps)
            return grad
        
        if best_result is not None:
            x_init = best_result
        else:
            x_init = x0
        
        try:
            result = minimize(
                objective_refine,
                x_init,
                method='L-BFGS-B',
                jac=jac_refine,
                bounds=[(0.1, 100.0)] * 12,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_c4:
                best_c4 = result.fun
                best_result = result.x
        except Exception as e:
            pass
    
    if best_result is not None and _validate_roots(best_result):
        return [float(r) for r in best_result]
    else:
        # Fallback: return the Laguerre roots or default
        if _validate_roots(x0):
            return [float(r) for r in x0]
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END

