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
    2. Try quadratic reconstruction (3 evals + 1 verify = 4 calls).
    3. Try cubic reconstruction (4 evals + 1 verify = 5 calls).
    4. If both fail, fall back to bracketing + Brent's method.
    """
    TOL = 1e-8
    DEADLINE = 110.0
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # Initial check
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
    
    # Points for interpolation
    # Use a step size that balances numerical stability and range.
    # If x0 is 0, h=1. If x0 is large, h should be relative?
    # For polynomials, relative step might be better if coefficients are large.
    # But usually fixed step works well for "smooth" standard test functions.
    # Let's try h = 1.0. If x0 is very large, 1.0 might be too small for precision?
    # np.vander handles it if we normalize? No, we are solving directly.
    # Let's stick to h=1.0 for simplicity, but check for non-finite results.
    
    h = 1.0
    x_minus = x0 - h
    x_plus = x0 + h
    
    # Store evaluations for reuse
    f_vals = {x0: fx0}
    x_vals = [x0]
    
    # Strategy 1: Quadratic Reconstruction
    # Evaluate at x0 - h and x0 + h
    try:
        f_minus = f(x_minus)
        f_plus = f(x_plus)
        f_vals[x_minus] = f_minus
        f_vals[x_plus] = f_plus
        x_vals.extend([x_minus, x_plus])
        
        if is_finite(f_minus) and is_finite(f_plus):
            # Solve for a*x^2 + b*x + c
            # Points: x0, x_minus, x_plus
            # Use np.polyfit or solve Vandermonde
            # Vandermonde for degree 2: [1, x, x^2]
            X = np.array([x_vals])
            V = np.vander(X, N=3, increasing=True).T # Shape (3, 3)
            # Actually, vander with increasing=True gives [1, x, x^2]
            # We have 3 points.
            Y = np.array([f_vals[x] for x in x_vals])
            
            try:
                coeffs = np.linalg.solve(V, Y)
                # coeffs = [c, b, a]
                a, b, c = coeffs[2], coeffs[1], coeffs[0]
                
                if abs(a) > 1e-12:
                    # Roots of ax^2 + bx + c = 0
                    disc = b*b - 4*a*c
                    if disc >= 0:
                        roots = [(-b + math.sqrt(disc)) / (2*a), 
                                 (-b - math.sqrt(disc)) / (2*a)]
                    else:
                        roots = []
                else:
                    if abs(b) > 1e-12:
                        roots = [-c / b]
                    else:
                        roots = []
                
                if roots:
                    best_r = min(roots, key=lambda r: abs(r - x0))
                    if is_finite(best_r):
                        f_best = f(best_r)
                        if is_finite(f_best) and abs(f_best) <= TOL:
                            return best_r
            except np.linalg.LinAlgError:
                pass
    except Exception:
        pass

    # Strategy 2: Cubic Reconstruction
    # We already have x0, x0-h, x0+h.
    # Add x0 + 2h
    x_extra = x0 + 2 * h
    try:
        f_extra = f(x_extra)
        f_vals[x_extra] = f_extra
        x_vals.append(x_extra)
        
        if is_finite(f_extra):
            # Solve for a*x^3 + b*x^2 + c*x + d
            X = np.array(x_vals)
            V = np.vander(X, N=4, increasing=True).T # Shape (4, 4)
            Y = np.array([f_vals[x] for x in x_vals])
            
            try:
                coeffs = np.linalg.solve(V, Y)
                # coeffs = [d, c, b, a]
                a, b, c, d = coeffs[3], coeffs[2], coeffs[1], coeffs[0]
                
                roots = np.roots([a, b, c, d])
                
                real_roots = []
                for r in roots:
                    if abs(r.imag) < 1e-10:
                        real_roots.append(r.real)
                
                if real_roots:
                    best_r = min(real_roots, key=lambda r: abs(r - x0))
                    if is_finite(best_r):
                        f_best = f(best_r)
                        if is_finite(f_best) and abs(f_best) <= TOL:
                            return best_r
            except np.linalg.LinAlgError:
                pass
    except Exception:
        pass

    # Strategy 3: Bracketing + Brent's Method (Fallback)
    step = 1.0
    # Reuse known values if possible?
    # We know f at x0, x0-h, x0+h, x0+2h.
    # Let's just do standard expansion from x0.
    
    for i in range(100):
        if not check_deadline():
            break
            
        a = x0 - step
        b = x0 + step
        
        # Check if we already evaluated these
        if a in f_vals:
            fa = f_vals[a]
        else:
            try:
                fa = f(a)
                f_vals[a] = fa
            except Exception:
                break
        if b in f_vals:
            fb = f_vals[b]
        else:
            try:
                fb = f(b)
                f_vals[b] = fb
            except Exception:
                break
            
        if not (is_finite(fa) and is_finite(fb)):
            break
            
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
            try:
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                break
                
        step *= 2.0
        if step > 1e10:
            break

    # Strategy 4: Safeguarded Newton / Secant
    x = x0
    fx = fx0
    
    for i in range(50):
        if not check_deadline():
            break
            
        if abs(fx) <= TOL:
            return x
            
        h = 1e-5 * max(1.0, abs(x))
        
        # Use cached values if available?
        # Secant needs two points. We have many.
        # Let's pick the two closest points to x in f_vals
        # Or just use central difference for derivative if Newton
        # But we can use Secant with best two points.
        
        # Find two points in f_vals with distinct x
        if len(f_vals) >= 2:
            keys = list(f_vals.keys())
            # Sort by distance to x
            keys.sort(key=lambda k: abs(k - x))
            x1 = keys[0]
            x2 = keys[1]
            f1 = f_vals[x1]
            f2 = f_vals[x2]
            
            if abs(x2 - x1) > 1e-15:
                slope = (f2 - f1) / (x2 - x1)
                if abs(slope) > 1e-15:
                    x_new = x - fx / slope
                    if is_finite(x_new):
                        try:
                            fx_new = f(x_new)
                        except Exception:
                            break
                        if is_finite(fx_new):
                            f_vals[x_new] = fx_new
                            if abs(fx_new) < best_f_abs:
                                best_f_abs = abs(fx_new)
                                best_x = x_new
                            if abs(fx_new) <= TOL:
                                return x_new
                            if abs(fx_new) < abs(fx):
                                x = x_new
                                fx = fx_new
                                continue
        # Fallback to central diff if secant fails or not enough points
        try:
            fx_h = f(x + h)
            fx_mh = f(x - h)
        except Exception:
            break
            
        if not (is_finite(fx_h) and is_finite(fx_mh)):
            break
            
        f_vals[x+h] = fx_h
        f_vals[x-h] = fx_mh
        
        dfx = (fx_h - fx_mh) / (2 * h)
        
        if abs(dfx) < 1e-15:
            break
            
        step = fx / dfx
        x_new = x - step
        
        if not is_finite(x_new) or abs(step) > 10 * max(1.0, abs(x)):
            break
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not is_finite(fx_new):
            break
            
        f_vals[x_new] = fx_new
        if abs(fx_new) < best_f_abs:
            best_f_abs = abs(fx_new)
            best_x = x_new
            
        if abs(fx_new) <= TOL:
            return x_new
        if abs(fx_new) > abs(fx) * 1.1 and i > 2:
            break
            
        x = x_new
        fx = fx_new

    return best_x
# EVOLVE-BLOCK-END
