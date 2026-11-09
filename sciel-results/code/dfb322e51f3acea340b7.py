"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 620.0  # Safety margin before 630s kill

def _compute_c4(roots):
    """Compute the C4 upper bound given 12 positive double-root locations."""
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    # P(x) = prod_{i=1}^{12} (x - r_i)^2
    # C4 bound: P(0) / integral_0^inf P(x) * w(x) dx where w is weight
    # For Laguerre weight: w(x) = exp(-x)
    # P(0) = prod r_i^2
    p0 = np.prod(roots ** 2)
    
    # Numerically compute integral of P(x) * exp(-x) from 0 to inf
    # Use Gauss-Laguerre quadrature with higher precision
    x, w = np.polynomial.laguerre.laggauss(300)
    p_vals = np.prod((x[:, None] - roots[None, :]) ** 2, axis=1)
    integral = np.sum(w * p_vals * np.exp(-x))
    
    if integral <= 0 or not np.isfinite(integral):
        return 1e10
    
    c4 = p0 / integral
    return c4

def _compute_c4_log(y):
    """Compute C4 in log-space for numerical stability."""
    roots = np.exp(y)
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        return 1e10
    
    p0 = np.prod(roots ** 2)
    x, w = np.polynomial.laguerre.laggauss(300)
    p_vals = np.prod((x[:, None] - roots[None, :]) ** 2, axis=1)
    integral = np.sum(w * p_vals * np.exp(-x))
    
    if integral <= 0 or not np.isfinite(integral):
        return 1e10
    
    c4 = p0 / integral
    return c4

def construct():
    """Return 12 positive double-root locations minimizing C4."""
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 30:
        return [4.0 * (i + 1) for i in range(12)]
    
    best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    best_c4 = _compute_c4(best_roots)
    
    try:
        # Phase 1: L-BFGS-B with log parameterization
        def _objective_lbfgs(y):
            return _compute_c4_log(y)
        
        y0 = np.log(best_roots)
        
        n_restarts = 8
        for r in range(n_restarts):
            if time.time() - _start_time > _DEADLINE - 120:
                break
            
            if r == 0:
                y_init = y0.copy()
            elif r == 1:
                # Try tighter spacing
                y_init = np.log(np.linspace(1.0, 20.0, 12))
            elif r == 2:
                # Try wider spacing
                y_init = np.log(np.linspace(2.0, 40.0, 12))
            else:
                np.random.seed(42 + r)
                y_init = y0 + np.random.randn(12) * 0.3
            
            result = minimize(_objective_lbfgs, y_init, method='L-BFGS-B', 
                            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < best_c4:
                roots = np.exp(result.x)
                if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                    c4 = _compute_c4(roots)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = roots.copy()
        
        # Phase 2: SLSQP with explicit bounds for refinement
        if time.time() - _start_time < _DEADLINE - 60:
            def _objective_slsqp(roots):
                if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
                    return 1e10
                return _compute_c4(roots)
            
            # Bounds: roots must be positive and increasing
            # Use log-space for SLSQP too
            def _objective_slsqp_log(y):
                roots = np.exp(y)
                if np.any(np.diff(roots) <= 0):
                    return 1e10
                return _compute_c4(roots)
            
            y_best = np.log(best_roots)
            result = minimize(_objective_slsqp_log, y_best, method='SLSQP',
                            options={'maxiter': 200, 'ftol': 1e-12})
            
            if result.fun < best_c4:
                roots = np.exp(result.x)
                if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                    c4 = _compute_c4(roots)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = roots.copy()
        
        # Phase 3: Final high-precision polish
        if time.time() - _start_time < _DEADLINE - 30:
            y_best = np.log(best_roots)
            result = minimize(_objective_lbfgs, y_best, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if result.fun < best_c4:
                roots = np.exp(result.x)
                if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                    c4 = _compute_c4(roots)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = roots.copy()
    
    except Exception:
        pass
    
    return list(best_roots)
# EVOLVE-BLOCK-END
