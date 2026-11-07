"""Optimized search for coefficients c0, c1, c2 minimizing r^2/(2*pi)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard: evaluator kills after 210s, keep 30s safety margin
_DEADLINE = 180.0
_t_start = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _t_start)

# Precompute Hermite polynomial coefficients
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

# Values at x=0:
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0

def _q_coefficients(c0, c1, c2):
    """
    Return coefficients [a0, a1, a2, a3, a4, a5] of q(y) where y = x^2.
    q(y) = g(x)/x^2 with g = c0*H0 + c1*H4 + c2*H8 + c3*H12.
    g(x) = sum_{k=0}^6 b_k x^{2k}. q(y) = sum_{k=0}^5 b_{k+1} y^k.
    """
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Coefficients of g(x) for powers x^2, x^4, ..., x^12
    # x^2: -48*c1 - 1344*c2 - 921600*c3
    # x^4: 16*c1 + 2688*c2 + 1424640*c3
    # x^6: -1344*c2 - 1003520*c3
    # x^8: 256*c2 + 332160*c3
    # x^10: -46080*c3
    # x^12: 4096*c3
    
    a0 = -48.0 * c1 - 1344.0 * c2 - 921600.0 * c3
    a1 = 16.0 * c1 + 2688.0 * c2 + 1424640.0 * c3
    a2 = -1344.0 * c2 - 1003520.0 * c3
    a3 = 256.0 * c2 + 332160.0 * c3
    a4 = -46080.0 * c3
    a5 = 4096.0 * c3
    
    return np.array([a0, a1, a2, a3, a4, a5])

def _find_largest_sign_change_root(coeffs):
    """
    Find the largest real root of the polynomial (in y) at which it changes sign.
    Returns the root r (in y-space).
    If no such root exists, return -inf.
    """
    # np.roots expects highest degree first
    poly_coeffs = coeffs[::-1]
    roots = np.roots(poly_coeffs)
    
    # Filter real roots > 0
    real_roots = []
    for root in roots:
        if np.isreal(root) and root.real > 1e-9:
            real_roots.append(root.real)
            
    if not real_roots:
        return -np.inf
        
    real_roots.sort(reverse=True)
    
    # Evaluate polynomial at points around the root to check sign change
    # Use a small epsilon relative to the root magnitude
    def eval_poly(y):
        return np.polyval(poly_coeffs, y)
        
    for root in real_roots:
        eps = 1e-6 * max(1.0, abs(root))
        val_minus = eval_poly(root - eps)
        val_plus = eval_poly(root + eps)
        
        # Check for sign change
        if val_minus * val_plus < 0:
            return root
            
    return -np.inf

def _evaluate_value(c0, c1, c2):
    """
    Compute the value = r / (2*pi) where r is the largest sign-changing root of q(y).
    Note: The task says value = r^2 / (2*pi) but defines r as the root of q(y) where y=x^2.
    Wait, let's re-read carefully.
    "Let q = g / x^2... Let r be the largest real root of q... value = r^2 / (2 * pi)"
    If q is a function of x, r is a root in x-space.
    But q = g/x^2. If g is even, q is even.
    The inspirations treated q as a polynomial in y=x^2.
    If r is the root in x-space, then y = r^2 is the root in y-space.
    The formula value = r^2 / (2*pi) implies r is the x-coordinate.
    If we find the root Y in y-space, then r = sqrt(Y).
    So value = Y / (2*pi).
    Let's verify.
    If q(x) = 0 at x=r, then q(sqrt(Y)) = 0 at Y=r^2.
    So if we find the largest Y, value = Y / (2*pi).
    """
    coeffs = _q_coefficients(c0, c1, c2)
    
    # Check if g tends to negative limit at infinity
    # Leading term of g is c3 * 4096 * x^12.
    # If c3 < 0, g -> -inf. Replace g by -g.
    # This negates all coeffs of q.
    # Roots remain the same. Sign change property remains the same.
    # So we don't need to explicitly handle this for the value calculation,
    # as long as we correctly identify the sign-changing roots.
    
    Y = _find_largest_sign_change_root(coeffs)
    
    if Y < 0:
        # No valid root found. This is bad.
        # Return a large value to indicate poor performance
        return 1e10
        
    return Y / (2.0 * np.pi)

def _objective(params):
    """Objective function for optimization: minimize the value."""
    c0, c1, c2 = params
    try:
        val = _evaluate_value(c0, c1, c2)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    """Search for optimal coefficients c0, c1, c2."""
    start_time = time.time()
    deadline = start_time + 180  # 180 seconds total, leave margin
    
    # Seed solution
    seed = [-113.0/100.0, 1.0/25.0, 1.0/3240.0]
    best_params = np.array(seed)
    best_value = _objective(best_params)
    
    # Strategy 1: Differential evolution with bounded search
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            # Bounds based on seed magnitudes, expanded
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-0.5, 0.5)]
            result = differential_evolution(
                _objective, bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            pass
    
    # Strategy 2: Local refinement with Nelder-Mead from best point
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            nm_result = minimize(
                _objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_result.fun < best_value:
                best_params = nm_result.x
                best_value = nm_result.fun
        except Exception:
            pass
            
    # Strategy 3: Perturb and refine from multiple starting points
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        try:
            rng = np.random.RandomState(123)
            for _ in range(20):
                if time.time() > deadline - 15:
                    break
                scale = 0.1 + 0.2 * np.random.rand()
                perturb = best_params + rng.normal(0, scale, 3)
                local_result = minimize(
                    _objective, perturb,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                if local_result.fun < best_value:
                    best_params = local_result.x
                    best_value = local_result.fun
        except Exception:
            pass
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            final_result = minimize(
                _objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if final_result.fun < best_value:
                best_params = final_result.x
                best_value = final_result.fun
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
