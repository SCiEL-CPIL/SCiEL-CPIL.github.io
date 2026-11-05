import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_chebyt

N = 400
DEADLINE = 145.0  # 5s safety margin before 150s kill

# Precompute Chebyshev basis on [-1, 1]
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

def _spectral_guided_init(degree, seed):
    """Generate initial coefficient vectors guided by spectral analysis."""
    rng = np.random.RandomState(seed)
    inits = []
    # Try several different spectral patterns
    for trial in range(5):
        coeffs = np.zeros(degree + 1)
        coeffs[0] = 1.0  # DC component
        # Add some frequency components
        for k in range(1, min(degree + 1, 10)):
            # Weight higher frequencies less
            coeffs[k] = rng.randn() / (1.0 + 0.5 * k)
        inits.append(coeffs)
    return inits

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Phase 1: Global search via differential evolution in low-dim Chebyshev space
    # Try multiple degrees with increasing complexity
    degrees_to_try = [4, 6, 8, 10, 12]
    
    for degree in degrees_to_try:
        if time_left() < 25:
            break
        
        B = _get_basis(degree)
        num_coeffs = degree + 1
        bounds = [(-3.0, 3.0)] * num_coeffs
        
        # Use differential evolution with moderate population for speed
        try:
            result = differential_evolution(
                _objective_full,
                bounds,
                seed=42 + degree,
                maxiter=40,
                popsize=12,
                tol=1e-5,
                init='sobol',
                polish=True,
                updating='deferred',
                mutation=(0.3, 0.9),
                recombination=0.8
            )
            if time_left() < 20:
                break
            
            h_candidate = _h_from_coeffs(result.x, degree)
            v = _value(h_candidate)
            if v < best_val:
                best_val = v
                best_h = h_candidate.copy()
                
        except Exception:
            pass
    
    # Phase 2: Coarse-to-fine grid refinement around best solution
    if time_left() > 15:
        # Find the best degree from Phase 1 by re-evaluating
        best_degree = 4
        best_coeffs = None
        for degree in degrees_to_try:
            if time_left() < 15:
                break
            # Quick local optimization to find good coefficients for this degree
            x0 = np.zeros(degree + 1)
            x0[0] = 1.0
            try:
                res = minimize(
                    _objective_full,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * (degree + 1),
                    options={'maxiter': 100, 'maxfun': 200}
                )
                v = _value(_h_from_coeffs(res.x, degree))
                if v < best_val:
                    best_val = v
                    best_h = _h_from_coeffs(res.x, degree).copy()
                    best_degree = degree
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Full-space gradient refinement from best low-dim solution
    if time_left() > 10:
        h0 = best_h.copy()
        
        def _full_obj(h):
            return _value(h)
        
        # Use L-BFGS-B with limited iterations
        try:
            res = minimize(
                _full_obj,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 30, 'maxfun': 100, 'ftol': 1e-8}
            )
            if time_left() > 5:
                h_candidate = res.x
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Phase 4: Try structured patterns as additional candidates
    if time_left() > 8:
        x = np.arange(N)
        # Sine/cosine patterns with various frequencies
        for k in [1, 2, 3, 5, 7, 11, 13, 17, 23, 31]:
            if time_left() < 5:
                break
            for amp in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
                if time_left() < 3:
                    break
                h = 1.0 + amp * np.cos(2 * np.pi * k * x / N)
                s = np.sum(h)
                if abs(s) < 1e-12:
                    continue
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Phase 5: Final local polish from best solution
    if time_left() > 5:
        h0 = best_h.copy()
        try:
            def _polish_obj(h):
                return _value(h)
            
            res = minimize(
                _polish_obj,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if time_left() > 2:
                h_candidate = res.x
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
