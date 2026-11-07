"""Root finder for 3 fixed smooth polynomial equations."""
import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Uses a hybrid strategy:
    1. Try Newton/secant with bracketing for fast convergence.
    2. If that fails, fall back to a robust bracketing search.
    3. Final polish with tight tolerance.
    Time guard to ensure we finish well before the 150s limit.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s evaluator limit (and 150s kill)
    
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return x
    
    # Phase 1: Try to establish a bracket and use secant/bisection hybrid
    # First, try to find a bracket around x0 by expanding
    a, b = None, None
    fa, fb = None, None
    
    # Check if x0 itself is a root
    if abs(best_fx) <= 1e-8:
        return x
    
    # Try to find a bracket by expanding from x0
    step = max(1.0, abs(x0)) * 0.1
    found_bracket = False
    
    for i in range(50):
        if time.time() > deadline:
            break
        # Try x0 - step and x0 + step
        x_left = x0 - step
        x_right = x0 + step
        
        fx_left = f(x_left)
        fx_right = f(x_right)
        
        # Check if x_left is a root
        if abs(fx_left) <= 1e-8:
            return x_left
        # Check if x_right is a root
        if abs(fx_right) <= 1e-8:
            return x_right
        
        # Check for sign change between x_left and x0
        if fx_left * best_fx < 0:
            a, b = x_left, x
            fa, fb = fx_left, best_fx
            found_bracket = True
            break
        
        # Check for sign change between x0 and x_right
        if best_fx * fx_right < 0:
            a, b = x, x_right
            fa, fb = best_fx, fx_right
            found_bracket = True
            break
        
        # Expand step
        step *= 2.0
    
    if not found_bracket:
        # No bracket found near x0. Try a wider search.
        # For polynomials, roots are finite. Try sampling over a wider range.
        # But first, let's try Newton's method from x0 without bracketing,
        # with damping to avoid divergence.
        x_newton = x
        for i in range(30):
            if time.time() > deadline:
                break
            fx = f(x_newton)
            if abs(fx) <= 1e-8:
                return x_newton
            
            # Finite difference derivative
            h = 1e-7 * max(1.0, abs(x_newton))
            fx_h = f(x_newton + h)
            d = (fx_h - fx) / h
            
            if d == 0 or not math.isfinite(d):
                break
            
            dx = fx / d
            # Damp the step if it's too large
            if abs(dx) > 10 * max(1.0, abs(x_newton)):
                dx = 10 * max(1.0, abs(x_newton)) * math.copysign(1, dx)
            
            x_newton -= dx
            
            if not math.isfinite(x_newton):
                break
            
            fx_new = f(x_newton)
            if abs(fx_new) < best_abs:
                best_x = x_newton
                best_abs = abs(fx_new)
                if best_abs <= 1e-8:
                    return best_x
        
        if best_abs <= 1e-8:
            return best_x
        
        # If Newton didn't work, try a broader search for a bracket
        # Sample at multiple points
        base = max(1.0, abs(x0))
        candidates = []
        for mult in [0.5, 1, 2, 5, 10, 50, 100, 500, 1000]:
            for sign in [-1, 1]:
                candidates.append(sign * base * mult)
        # Also try small perturbations
        for sign in [-1, 1]:
            candidates.append(x0 + sign * base * 0.1)
            candidates.append(x0 + sign * base * 0.5)
        
        prev_x = None
        prev_fx = None
        for cx in candidates:
            if time.time() > deadline:
                break
            try:
                cfx = f(cx)
            except:
                continue
            if abs(cfx) <= 1e-8:
                return cx
            if prev_x is not None and prev_fx is not None:
                if prev_fx * cfx < 0:
                    a, b = prev_x, cx
                    fa, fb = prev_fx, cfx
                    found_bracket = True
                    break
            prev_x = cx
            prev_fx = cfx
        
        if not found_bracket:
            # Last resort: return best found
            if best_abs <= 1e-8:
                return best_x
            # Try one more thing: secant from two nearby points
            h = 1e-4 * max(1.0, abs(x0))
            x1 = x0
            x2 = x0 + h
            f1 = f(x1)
            f2 = f(x2)
            if f1 * f2 < 0:
                a, b = x1, x2
                fa, fb = f1, f2
                found_bracket = True
            else:
                # Return best effort
                return best_x
    
    # Phase 2: Use bisection + secant hybrid on the bracket [a, b]
    if found_bracket:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Verify bracket
        if fa * fb > 0:
            # No valid bracket, return best
            return best_x
        
        # Bisection with secant acceleration
        for i in range(100):
            if time.time() > deadline:
                break
            
            mid = (a + b) / 2.0
            fmid = f(mid)
            
            if abs(fmid) <= 1e-8:
                return mid
            
            # Update best
            if abs(fmid) < best_abs:
                best_x = mid
                best_abs = abs(fmid)
            
            # Narrow bracket
            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
            
            # Try secant step
            if fa * fb < 0 and fb != fa:
                # Secant formula: x_new = b - fb*(b-a)/(fb-fa)
                x_sec = b - fb * (b - a) / (fb - fa)
                if a < x_sec < b:
                    fx_sec = f(x_sec)
                    if abs(fx_sec) <= 1e-8:
                        return x_sec
                    if abs(fx_sec) < best_abs:
                        best_x = x_sec
                        best_abs = abs(fx_sec)
                    if fa * fx_sec < 0:
                        b = x_sec
                        fb = fx_sec
                    else:
                        a = x_sec
                        fa = fx_sec
                    if best_abs <= 1e-8:
                        return best_x

            # Newton polish if bracket is small
            if abs(b - a) < 1e-5 * max(1.0, abs(mid)):
                # Estimate derivative at mid
                h = 1e-7 * max(1.0, abs(mid))
                f_mid_h = f(mid + h)
                d = (f_mid_h - fmid) / h
                if d != 0 and math.isfinite(d):
                    x_new = mid - fmid / d
                    if a < x_new < b:
                        f_new = f(x_new)
                        if abs(f_new) <= 1e-8:
                            return x_new
                        if abs(f_new) < best_abs:
                            best_x = x_new
                            best_abs = abs(f_new)
                        # Update bracket
                        if fa * f_new < 0:
                            b = x_new
                            fb = f_new
                        else:
                            a = x_new
                            fa = f_new
                        if best_abs <= 1e-8:
                            return best_x

            # Convergence check
            if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
                break
        
        return best_x
    
    # No bracket found, return best effort
    return best_x

# EVOLVE-BLOCK-END
