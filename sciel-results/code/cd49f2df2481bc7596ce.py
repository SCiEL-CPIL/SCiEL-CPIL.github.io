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
    2. Attempt cubic interpolation (degree 3) using 4 points if possible.
       - This exploits the 'polynomial' hint. If the function is a cubic,
         we can solve it exactly in 4-5 calls.
       - If not cubic, the residual will be non-zero, and we fall back.
    3. Attempt Newton-Raphson iterations.
       - Use central difference for derivative.
       - Accept step only if it reduces |f(x)|.
       - If a bracket is formed (sign change), switch to Brent's method.
    4. Fallback to bracketing search + Brent's method if Newton fails.
    Includes a deadline guard to prevent timeout.
    """
    TOL = 1e-8
    DEADLINE = 115.0  # Safety margin before 150s kill
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

    # --- Strategy 1: Cubic Interpolation Attempt ---
    # If f is a polynomial of degree <= 3, we can determine it exactly
    # using 4 points. Let's try to fit a cubic to 4 points around x0.
    # Points: x0, x0+h, x0+2h, x0+3h
    # We need to choose h. A small h is good for local approximation,
    # but for exact polynomial identification, any non-zero h works
    # provided the points are distinct and we solve the system.
    # However, if the polynomial is high degree, this will fail (residual > tol).
    
    # Let's try a few scales for h to increase robustness.
    h_base = max(1e-3, 1e-3 * abs(x0))
    
    cubic_found = False
    for scale in [1.0, 10.0, 0.1, 100.0]:
        if not check_deadline():
            break
            
        h = h_base * scale
        
        # Define points
        x1 = x0
        x2 = x0 + h
        x3 = x0 + 2*h
        x4 = x0 + 3*h
        
        try:
            f1 = f(x1)
            f2 = f(x2)
            f3 = f(x3)
            f4 = f(x4)
        except Exception:
            continue
            
        if not all(is_finite(v) for v in [f1, f2, f3, f4]):
            continue
            
        # Update best
        for xi, fi in [(x1, f1), (x2, f2), (x3, f3), (x4, f4)]:
            if abs(fi) < best_f_abs:
                best_f_abs = abs(fi)
                best_x = xi
                
        if abs(f1) <= TOL: return x1
        if abs(f2) <= TOL: return x2
        if abs(f3) <= TOL: return x3
        if abs(f4) <= TOL: return x4

        # Construct Vandermonde-like system for cubic coefficients
        # P(x) = a*x^3 + b*x^2 + c*x + d
        # We solve for a,b,c,d using the 4 points.
        # To avoid ill-conditioning, we can shift coordinates to center at x0.
        # Let t = x - x0. Then t values are 0, h, 2h, 3h.
        # P(x0+t) = A*t^3 + B*t^2 + C*t + D
        # D = f1
        # A*h^3 + B*h^2 + C*h + D = f2
        # A*(2h)^3 + B*(2h)^2 + C*(2h) + D = f3
        # A*(3h)^3 + B*(3h)^2 + C*(3h) + D = f4
        
        # This is a linear system for A, B, C.
        # [h^3, h^2, h] [A]   [f2 - f1]
        # [8h^3, 4h^2, 2h] [B] = [f3 - f1]
        # [27h^3, 9h^2, 3h] [C]   [f4 - f1]
        
        try:
            M = np.array([
                [h**3, h**2, h],
                [(2*h)**3, (2*h)**2, 2*h],
                [(3*h)**3, (3*h)**2, 3*h]
            ])
            Y = np.array([f2 - f1, f3 - f1, f4 - f1])
            
            # Check condition number? Or just solve.
            coeffs = np.linalg.solve(M, Y)
            A, B, C = coeffs
            D = f1
            
            # The polynomial in t is P(t) = A*t^3 + B*t^2 + C*t + D
            # We want to find roots of this cubic.
            # np.roots expects coefficients from highest power to constant.
            roots_t = np.roots([A, B, C, D])
            
            # Filter real roots
            real_roots_t = [r.real for r in roots_t if abs(r.imag) < 1e-9]
            
            # For each real root, x = x0 + t
            # Check if it satisfies the original function
            for t in real_roots_t:
                x_candidate = x0 + t
                if not is_finite(x_candidate):
                    continue
                try:
                    f_candidate = f(x_candidate)
                except Exception:
                    continue
                if not is_finite(f_candidate):
                    continue
                if abs(f_candidate) <= TOL:
                    return x_candidate
                # If close, maybe refine?
                if abs(f_candidate) < best_f_abs:
                    best_f_abs = abs(f_candidate)
                    best_x = x_candidate
                    
            # If we didn't return, the cubic fit was likely inaccurate (higher degree poly)
            # or the root is complex/unstable.
            # We don't break immediately because a different h might yield a better local fit
            # or a valid root if the previous one was numerically unstable.
            # But if we found a root with small residual, we might have returned.
            
        except np.linalg.LinAlgError:
            continue
        except Exception:
            continue

    # --- Strategy 2: Newton-Raphson ---
    # We track a potential bracket [a, b] if signs differ
    a = x0
    fa = fx0
    b = None
    fb = None
    
    x = x0
    fx = fx0
    
    # Max Newton iterations to try before falling back
    max_newton_iters = 20
    
    for i in range(max_newton_iters):
        if not check_deadline():
            break
            
        if abs(fx) <= TOL:
            return x
            
        # Estimate derivative using central differences
        h = 1e-5 * max(1.0, abs(x))
        
        try:
            fx_h = f(x + h)
            fx_mh = f(x - h)
        except Exception:
            break
            
        if not (is_finite(fx_h) and is_finite(fx_mh)):
            break
            
        dfx = (fx_h - fx_mh) / (2 * h)
        
        if abs(dfx) < 1e-15:
            # Derivative zero, Newton fails
            break
            
        # Newton step
        step = fx / dfx
        x_new = x - step
        
        if not is_finite(x_new):
            break
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not is_finite(fx_new):
            break
            
        # Update best
        if abs(fx_new) < best_f_abs:
            best_f_abs = abs(fx_new)
            best_x = x_new
            
        # Check for convergence
        if abs(fx_new) <= TOL:
            return x_new
            
        # Check for bracket formation
        # If f(x) and f(x_new) have opposite signs, we have a bracket
        if fa * fx_new < 0:
            # Bracket is [a, x_new]
            a_bracket = a
            b_bracket = x_new
            try:
                root = scipy.optimize.brentq(f, a_bracket, b_bracket, xtol=1e-12, rtol=1e-12, maxiter=50)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                pass
        
        # If f(x_new) and f(x) have opposite signs (and we didn't catch it above because 'a' was updated?)
        # Actually, if fx * fx_new < 0, we have a bracket [x, x_new]
        if fx * fx_new < 0:
            a_bracket = x
            b_bracket = x_new
            try:
                root = scipy.optimize.brentq(f, a_bracket, b_bracket, xtol=1e-12, rtol=1e-12, maxiter=50)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                pass

        # Safeguard: Only accept the Newton step if it reduces the error
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            # Newton step did not improve.
            break

    # --- Strategy 3: Robust Bracketing Search ---
    # If Newton failed or wasn't used, we need to find a bracket.
    # We search around x0 (or best_x if it's better).
    center = best_x
    step = 1.0
    
    # Try to find a bracket [c - step, c + step]
    for i in range(100):
        if not check_deadline():
            break
            
        a = center - step
        b = center + step
        
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            break
            
        if not (is_finite(fa) and is_finite(fb)):
            break
            
        # Update best
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
            # Bracket found!
            try:
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                break
                
        # Expand step
        step *= 2.0
        
        # Safety limit
        if step > 1e10:
            break

    # If we reach here, we failed to find a root.
    # Return the best x found.
    return best_x
# EVOLVE-BLOCK-END
