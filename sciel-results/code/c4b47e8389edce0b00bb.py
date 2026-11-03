"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for the Laguerre polynomial combination
    with given double-root locations.
    
    For the Cohn-Goncalves construction with m=12, n=1:
    We form a polynomial P(x) = prod_{i=1}^{12} (x - r_i)^2
    The C4 bound is related to the integral of P(x) * L_12^(1)(x) * x^1 * e^{-x} dx
    normalized appropriately.
    
    A practical computation: the upper bound on C4 involves the ratio of
    certain Laguerre moments. We compute it via quadrature.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Build the polynomial with double roots
    # P(x) = prod (x - r_i)^2
    # Use log-sum for stability
    def log_P(x):
        total = np.zeros_like(x)
        for r in roots:
            total += 2.0 * np.log(np.abs(x - r) + 1e-300)
        return total
    
    # The C4 bound for sign uncertainty with Laguerre weight
    # involves: C4 = (integral of P(x) * w(x) dx)^2 / (integral of P(x)^2 * w(x) dx) * const
    # where w(x) = x^n * e^{-x} for Laguerre n=1
    # Actually for the refined bound, it's more subtle.
    
    # Let's use a direct quadrature approach
    # The key quantity is the norm ratio
    # For double-root Laguerre combination, C4 ~ 1 / (sum of certain terms)
    
    # A simpler model: the C4 bound is inversely related to how well
    # the polynomial matches the Laguerre weight distribution.
    # We compute: C4 = 4 * (integral P(x) * x * e^{-x} dx)^2 / (integral P(x)^2 * x * e^{-x} dx)
    # This is a Rayleigh quotient type expression
    
    # Use Gauss-Laguerre quadrature
    n_quad = 200
    x_q, w_q = np.polynomial.laguerre.laggauss(n_quad)
    w_q = w_q * x_q  # weight x^1 * e^{-x}, but laggauss already includes e^{-x}
    
    # Evaluate P(x) at quadrature points
    # Use log to avoid overflow
    log_P_vals = np.zeros(n_quad)
    for r in roots:
        log_P_vals += 2.0 * np.log(np.abs(x_q - r) + 1e-300)
    
    # Handle sign: P(x) is always non-negative since it's product of squares
    P_vals = np.exp(log_P_vals)
    
    # Numerator: (integral P(x) * x * e^{-x} dx)^2
    num = np.sum(P_vals * w_q) ** 2
    
    # Denominator: integral P(x)^2 * x * e^{-x} dx
    den = np.sum(P_vals ** 2 * w_q)
    
    if den <= 0 or np.isinf(den) or np.isinf(num):
        return np.inf
    
    # C4 upper bound (proportional to the Rayleigh quotient)
    c4 = 4.0 * num / den
    
    return c4


def _objective(roots):
    """Objective function for optimization."""
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return 1e10
    return _compute_c4_bound(roots)


def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive floats.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Not enough time, return seed
        return [4.0 * (i + 1) for i in range(12)]
    
    # Initial guess: spread out roots
    # The seed uses [4, 8, 12, ..., 48]
    # Try clustering more optimally
    x0 = np.array([4.0 * (i + 1) for i in range(12)], dtype=np.float64)
    
    best_roots = x0.copy()
    best_val = _compute_c4_bound(best_roots)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        elapsed = time.time() - _start_time
        if elapsed < _DEADLINE - 10 and result.x is not None:
            val = _compute_c4_bound(result.x)
            if val < best_val and np.all(result.x > 0):
                best_roots = result.x
                best_val = val
    except Exception:
        pass
    
    elapsed = time.time() - _start_time
    if elapsed < _DEADLINE - 20:
        # Try a second optimization from a different starting point
        # Clustered initial guess
        x1 = np.linspace(1.0, 30.0, 12)
        try:
            result2 = minimize(
                _objective,
                x1,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            elapsed = time.time() - _start_time
            if elapsed < _DEADLINE - 10 and result2.x is not None:
                val2 = _compute_c4_bound(result2.x)
                if val2 < best_val and np.all(result2.x > 0):
                    best_roots = result2.x
                    best_val = val2
        except Exception:
            pass
    
    # Ensure all roots are positive
    best_roots = np.maximum(best_roots, 1e-6)
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
