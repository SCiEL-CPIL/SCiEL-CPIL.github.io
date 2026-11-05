"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_chebyt

N = 400
DEADLINE = 150.0

# Precompute Chebyshev basis on the interval
_x = np.linspace(-1, 1, N)
_basis_cache = {}

def _get_basis(degree):
    if degree in _basis_cache:
        return _basis_cache[degree]
    B = np.zeros((degree + 1, N))
    for k in range(degree + 1):
        B[k] = eval_chebyt(k, _x)
    _basis_cache[degree] = B
    return B

def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))

def _h_from_coeffs(coeffs, degree):
    B = _get_basis(degree)
    h = np.zeros(N)
    for i, c in enumerate(coeffs):
        h += c * B[i]
    return h

def _objective_full(coeffs):
    degree = len(coeffs) - 1
    h = _h_from_coeffs(coeffs, degree)
    return _value(h)

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    # Strategy 1: Low-dimensional Chebyshev basis search
    # Try different degrees and use differential evolution
    degrees_to_try = [3, 4, 5, 6, 7, 8, 10]
    
    for degree in degrees_to_try:
        if elapsed() > DEADLINE - 30:
            break
        
        B = _get_basis(degree)
        # Bounds: coefficients in [-5, 5]
        bounds = [(-5.0, 5.0)] * (degree + 1)
        
        # Use differential evolution for global search
        # Keep population small for speed
        try:
            result = differential_evolution(
                _objective_full,
                bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-4,
                init='sobol',
                polish=True,
                updating='deferred'
            )
            if elapsed() > DEADLINE - 25:
                break
            h_candidate = _h_from_coeffs(result.x, degree)
            v = _value(h_candidate)
            if v < best_val:
                best_val = v
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Strategy 2: Local refinement from best Chebyshev solution
    if elapsed() < DEADLINE - 20:
        # Use the best Chebyshev coefficients as starting point for local optimization
        # Convert to a parameterization and do L-BFGS-B
        try:
            # Find which degree gave best result by trying each
            best_coeffs = None
            best_deg = None
            for degree in degrees_to_try:
                if elapsed() > DEADLINE - 20:
                    break
                B = _get_basis(degree)
                # Do a quick local search
                x0 = np.zeros(degree + 1)
                # Initialize with small values
                x0[0] = 1.0  # constant term
                
                res = minimize(
                    _objective_full,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-10.0, 10.0)] * (degree + 1),
                    options={'maxiter': 200, 'maxfun': 500}
                )
                if elapsed() > DEADLINE - 15:
                    break
                h_candidate = _h_from_coeffs(res.x, degree)
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
                    best_coeffs = res.x.copy()
                    best_deg = degree
        except Exception:
            pass
    
    # Strategy 3: Try structured patterns that might be optimal
    # Theoretical: for a signal with large sum but small peak autocorrelation
    # Try: h[i] = 1 + a*cos(2*pi*i/N) + b*cos(4*pi*i/N) etc.
    if elapsed() < DEADLINE - 15:
        x = np.arange(N)
        for a in np.linspace(-2, 2, 9):
            for b in np.linspace(-1, 1, 7):
                if elapsed() > DEADLINE - 12:
                    break
                h = 1.0 + a * np.cos(2 * np.pi * x / N) + b * np.cos(4 * np.pi * x / N)
                s = np.sum(h)
                if abs(s) < 1e-12:
                    continue
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Strategy 4: Gradient-based refinement in full space from best solution
    if elapsed() < DEADLINE - 10:
        try:
            # Use the current best as starting point
            h0 = best_h.copy()
            
            def _full_objective(h):
                return _value(h)
            
            # Use Nelder-Mead or L-BFGS-B on the full 400-dim space
            # This is expensive, so limit iterations
            res = minimize(
                _full_objective,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 200}
            )
            if elapsed() < DEADLINE - 5:
                h_candidate = res.x
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final safety: ensure finite and non-zero sum
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
