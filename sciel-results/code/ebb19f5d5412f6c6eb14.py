import math
import time
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    
    Strategy:
    1. Check if x0 is already a root.
    2. Try to reconstruct the polynomial using interpolation at multiple point sets.
       Solve for roots analytically using numpy.roots.
       Select the root nearest to x0. Verify.
    3. If polynomial fit fails, fall back to bracketing + Brent's method.
    4. Final fallback: Safeguarded Newton method with line search.
    """
    TOL = 1e-8
    DEADLINE = 105.0
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # Initial check: is x0 already a root?
    try:
        fx0 = f(x0)
    except Exception:
        return x0
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_f_abs = abs(fx0)
    
    # Track all evaluations for potential reuse
    eval_cache = {x0: fx0}
    
    def f_cached(x):
        """Evaluate f(x) with caching."""
        if x in eval_cache:
            return eval_cache[x]
        try:
            val = f(x)
            if is_finite(val):
                eval_cache[x] = val
                return val
        except Exception:
            pass
        return float('nan')
    
    def find_best_root_from_coeffs(coeffs, x0_ref):
        """
        Given polynomial coefficients, find the real root nearest to x0_ref.
        """
        if coeffs is None or len(coeffs) == 0:
            return None
        
        try:
            roots = np.roots(coeffs)
        except Exception:
            return None
        
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                real_roots.append(r.real)
        
        if not real_roots:
            return None
        
        return min(real_roots, key=lambda r: abs(r - x0_ref))
    
    # STRATEGY 1: Polynomial reconstruction with multiple point sets
    # We try to fit polynomials of degree 2 and 3 at various configurations.
    
    point_sets = []
    
    # Symmetric points around x0 with varying spreads
    for h in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        # Set 1: x0-h, x0-h/2, x0+h/2, x0+h
        p1 = [x0 - h, x0 - h/2, x0 + h/2, x0 + h]
        if len(set(p1)) == 4:
            point_sets.append(p1)
        
        # Set 2: x0-1.5h, x0-0.5h, x0+0.5h, x0+1.5h
        p2 = [x0 - 1.5*h, x0 - 0.5*h, x0 + 0.5*h, x0 + 1.5*h]
        if len(set(p2)) == 4:
            point_sets.append(p2)
        
        # Set 3: Include x0
        p3 = [x0, x0 - h, x0 + h, x0 + 2*h]
        if len(set(p3)) == 4:
            point_sets.append(p3)
        
        # Set 4: Asymmetric
        p4 = [x0 - 2*h, x0 - h, x0, x0 + h]
        if len(set(p4)) == 4:
            point_sets.append(p4)

    for points in point_sets:
        if not check_deadline():
            break
        
        # Evaluate f at all points
        vals = [f_cached(p) for p in points]
        if not all(is_finite(v) for v in vals):
            continue
        
        # Check if any point is already a root
        for p, v in zip(points, vals):
            if abs(v) <= TOL:
                return p
            if abs(v) < best_f_abs:
                best_f_abs = abs(v)
                best_x = p
        
        # Try cubic fit (degree 3) using all 4 points
        try:
            coeffs_cubic = np.polyfit(np.array(points, dtype=float), np.array(vals, dtype=float), 3)
            candidate = find_best_root_from_coeffs(coeffs_cubic, x0)
            if candidate is not None and is_finite(candidate):
                fcand = f_cached(candidate)
                if is_finite(fcand):
                    if abs(fcand) <= TOL:
                        return candidate
                    if abs(fcand) < best_f_abs:
                        best_f_abs = abs(fcand)
                        best_x = candidate
        except Exception:
            pass
        
        # Try quadratic fit (degree 2) using first 3 points
        if len(points) >= 3:
            try:
                coeffs_quad = np.polyfit(np.array(points[:3], dtype=float), np.array(vals[:3], dtype=float), 2)
                candidate_quad = find_best_root_from_coeffs(coeffs_quad, x0)
                if candidate_quad is not None and is_finite(candidate_quad):
                    fcand_quad = f_cached(candidate_quad)
                    if is_finite(fcand_quad):
                        if abs(fcand_quad) <= TOL:
                            return candidate_quad
                        if abs(fcand_quad) < best_f_abs:
                            best_f_abs = abs(fcand_quad)
                            best_x = candidate_quad
            except Exception:
                pass

    # STRATEGY 2: Bracketing + Brent's method
    # Search for a bracket [a, b] where f(a)*f(b) < 0
    step = 1.0
    found_bracket = False
    
    for i in range(50):
        if not check_deadline():
            break
            
        a = x0 - step
        b = x0 + step
        
        fa = f_cached(a)
        fb = f_cached(b)
        
        if not (is_finite(fa) and is_finite(fb)):
            # Try smaller steps or break
            step *= 2.0
            if step > 1e12:
                break
            continue
            
        if abs(fa) < best_f_abs:
            best_f_abs = abs(fa)
            best_x = a
        if abs(fb) < best_f_abs:
            best_f_abs = abs(fb)
            best_x = b
            
        if fa == 0:
            return a
        if fb == 0:
            return b
            
        if fa * fb < 0:
            found_bracket = True
            try:
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if is_finite(root):
                    f_root = f_cached(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    # Even if not perfectly within TOL, return it if it's better
                    if is_finite(f_root) and abs(f_root) < best_f_abs:
                        best_f_abs = abs(f_root)
                        best_x = root
                    return root
            except Exception:
                pass
            break
            
        step *= 2.0
        if step > 1e12:
            break
    
    # STRATEGY 3: Safeguarded Newton method
    # Start from best_x found so far
    x = best_x if is_finite(best_x) else x0
    fx = f_cached(x)
    
    for i in range(30):
        if not check_deadline():
            break
            
        if abs(fx) <= TOL:
            return x
            
        # Estimate derivative using central difference
        h = 1e-5 * max(1.0, abs(x))
        fx_h = f_cached(x + h)
        fx_mh = f_cached(x - h)
        
        if not (is_finite(fx_h) and is_finite(fx_mh)):
            break
            
        dfx = (fx_h - fx_mh) / (2 * h)
        
        if abs(dfx) < 1e-15:
            break
            
        newton_step = fx / dfx
        x_new = x - newton_step
        
        if not is_finite(x_new) or abs(newton_step) > 10 * max(1.0, abs(x)):
            # Damping
            damp = 0.5
            improved = False
            for _ in range(10):
                x_damped = x - damp * newton_step
                if is_finite(x_damped):
                    fx_damped = f_cached(x_damped)
                    if is_finite(fx_damped):
                        if abs(fx_damped) < best_f_abs:
                            best_f_abs = abs(fx_damped)
                            best_x = x_damped
                        if abs(fx_damped) <= TOL:
                            return x_damped
                        if abs(fx_damped) < abs(fx):
                            x = x_damped
                            fx = fx_damped
                            improved = True
                            break
                damp *= 0.5
            if not improved:
                break
            continue
            
        fx_new = f_cached(x_new)
        
        if not is_finite(fx_new):
            break
            
        if abs(fx_new) < best_f_abs:
            best_f_abs = abs(fx_new)
            best_x = x_new
            
        if abs(fx_new) <= TOL:
            return x_new
            
        if abs(fx_new) > abs(fx) * 1.1 and i > 2:
            # Damping
            damp = 0.5
            improved = False
            for _ in range(10):
                x_damped = x - damp * (x_new - x)
                if is_finite(x_damped):
                    fx_damped = f_cached(x_damped)
                    if is_finite(fx_damped):
                        if abs(fx_damped) < abs(fx):
                            x = x_damped
                            fx = fx_damped
                            improved = True
                            break
                damp *= 0.5
            if not improved:
                break
            continue
            
        x = x_new
        fx = fx_new
    
    return best_x
# EVOLVE-BLOCK-END
