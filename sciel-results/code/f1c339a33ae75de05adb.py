"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 620.0  # Safety margin before 630s kill

def _compute_c4(roots):
    """Compute the C4 upper bound given 12 positive double-root locations.
    
    Uses log-space computation for numerical stability.
    C4 = P(0) / integral_0^inf P(x) * exp(-x) dx
    where P(x) = prod_{i=1}^{12} (x - r_i)^2
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    # P(0) = prod r_i^2, compute in log space
    log_p0 = 2.0 * np.sum(np.log(roots))
    
    # Numerically compute integral of P(x) * exp(-x) from 0 to inf
    # Use Gauss-Laguerre quadrature with more points for accuracy
    n_quad = 300
    x, w = np.polynomial.laguerre.laggauss(n_quad)
    
    diff = x[:, None] - roots[None, :]
    log_p_vals = 2.0 * np.sum(np.log(np.abs(diff)), axis=1)
    
    log_integrand = np.log(w) + log_p_vals - x
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return 1e10
    log_integral = max_log + np.log(np.sum(np.exp(log_integrand - max_log)))
    
    if not np.isfinite(log_integral):
        return 1e10
    
    log_c4 = log_p0 - log_integral
    if not np.isfinite(log_c4):
        return 1e10
    
    c4 = np.exp(log_c4)
    if not np.isfinite(c4) or c4 <= 0:
        return 1e10
    
    return c4

def _compute_c4_fast(roots):
    """Faster version using direct computation for smaller values."""
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    n_quad = 200
    x, w = np.polynomial.laguerre.laggauss(n_quad)
    
    diff = x[:, None] - roots[None, :]
    p_vals = np.prod(diff ** 2, axis=1)
    
    integrand = w * p_vals * np.exp(-x)
    integral = np.sum(integrand)
    
    if integral <= 0 or not np.isfinite(integral):
        return 1e10
    
    p0 = np.prod(roots ** 2)
    c4 = p0 / integral
    
    if not np.isfinite(c4) or c4 <= 0:
        return 1e10
    
    return c4

def construct():
    """Return 12 positive double-root locations minimizing C4."""
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 30:
        # Not enough time, return default
        return [4.0 * (i + 1) for i in range(12)]
    
    # Use the zeros of the Laguerre polynomial L_12(x) as the initial guess.
    # These are the Gauss-Laguerre quadrature nodes for n=12, which are the
    # roots of the orthogonal polynomial and provide a physically consistent
    # baseline for the Cohn-Goncalves construction.
    try:
        # laggauss(n) returns the nodes (roots of L_n) and weights
        # The nodes are the zeros of L_n(x)
        x_lag, w_lag = np.polynomial.laguerre.laggauss(12)
        best_roots = np.sort(x_lag)
    except Exception:
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    best_c4 = _compute_c4_fast(best_roots)
    
    # Try optimization
    try:
        # Use log-parameterization to ensure positivity
        def _log_transform(y):
            return np.exp(y)
        
        def _objective(y):
            roots = _log_transform(y)
            return _compute_c4_fast(roots)
        
        # Initial guess: log of Laguerre roots
        y0 = np.log(best_roots)
        
        # Multiple restarts with different strategies
        n_restarts = 12
        for r in range(n_restarts):
            if time.time() - _start_time > _DEADLINE - 90:
                break
            
            if r == 0:
                y_init = y0.copy()
            elif r == 1:
                # Small perturbation
                np.random.seed(100 + r)
                y_init = y0 + np.random.randn(12) * 0.1
            elif r == 2:
                # Medium perturbation
                np.random.seed(200 + r)
                y_init = y0 + np.random.randn(12) * 0.3
            elif r == 3:
                # Larger perturbation
                np.random.seed(300 + r)
                y_init = y0 + np.random.randn(12) * 0.5
            elif r == 4:
                # Compressed roots
                np.random.seed(400 + r)
                y_init = y0 + np.random.randn(12) * 0.2
                # Shift toward center
                y_init -= 0.1
            elif r == 5:
                # Expanded roots
                np.random.seed(500 + r)
                y_init = y0 + np.random.randn(12) * 0.2
                # Shift outward
                y_init += 0.1
            else:
                # Random perturbations with varying scales
                np.random.seed(600 + r)
                scale = 0.1 + 0.05 * (r - 6)
                y_init = y0 + np.random.randn(12) * max(0.05, scale)
            
            try:
                result = minimize(_objective, y_init, method='L-BFGS-B', 
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if result.fun < best_c4:
                    roots = _log_transform(result.x)
                    if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                        c4 = _compute_c4_fast(roots)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = roots.copy()
            except Exception:
                continue
            
            # Check time
            if time.time() - _start_time > _DEADLINE - 120:
                break
        
        # Final high-precision polish pass with tighter tolerances
        if time.time() - _start_time < _DEADLINE - 60:
            try:
                def _objective_precise(y):
                    roots = _log_transform(y)
                    return _compute_c4(roots)  # Use the more accurate version
                
                y_init = np.log(best_roots)
                result = minimize(_objective_precise, y_init, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if result.fun < best_c4:
                    roots = _log_transform(result.x)
                    if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                        c4 = _compute_c4(roots)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = roots.copy()
            except Exception:
                pass
    except Exception:
        pass
    
    return list(best_roots)
# EVOLVE-BLOCK-END
