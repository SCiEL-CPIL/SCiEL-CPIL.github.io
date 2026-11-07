import math
import time

def find_root(f, x0):
    TOL = 1e-8
    TOL_POLISH = 1e-12
    MAX_CALLS = 490
    deadline = time.time() + 145.0
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS or time.time() > deadline:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            a = abs(val)
            if a < best_abs:
                best_abs = a
                best_x = x
            return val
        except Exception:
            return float('inf')
    
    x0 = float(x0)
    
    # First, evaluate at x0
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    if not math.isfinite(fx0):
        # x0 is in a bad region, try nearby
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            x_try = x0 + dx
            fx_try = safe_f(x_try)
            if abs(fx_try) <= TOL:
                return x_try
            if math.isfinite(fx_try):
                best_x = x_try
                best_abs = abs(fx_try)
                break
        return best_x
    
    # Try to find a bracket near x0 using small steps
    # For oscillatory functions, the period is likely O(1) or O(10)
    # Try multiple scales of local search
    
    bracket_a = None
    bracket_b = None
    
    # Try progressively larger local windows
    for scale in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if calls >= MAX_CALLS - 30 or time.time() > deadline:
            break
        
        # Check left and right of x0
        xl = x0 - scale
        xr = x0 + scale
        
        fxl = safe_f(xl)
        fxr = safe_f(xr)
        
        if not math.isfinite(fxl) or not math.isfinite(fxr):
            continue
        
        # Check if sign change between xl and x0
        if fx0 * fxl < 0:
            bracket_a = xl
            bracket_b = x0
            break
        
        # Check if sign change between x0 and xr
        if fx0 * fxr < 0:
            bracket_a = x0
            bracket_b = xr
            break
        
        # Check if sign change between xl and xr (x0 might be outside)
        if fxl * fxr < 0:
            bracket_a = xl
            bracket_b = xr
            break
    
    if bracket_a is not None:
        # Use bisection + IQUO (inverse quadratic interpolation) hybrid to refine
        a, b = bracket_a, bracket_b
        fa = safe_f(a)
        fb = safe_f(b)
        
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            # Re-evaluate to be safe
            fa = safe_f(a)
            fb = safe_f(b)
        
        # Track previous point for IQUO
        x_prev = None
        f_prev = None
        
        for _ in range(60):
            if calls >= MAX_CALLS - 15 or time.time() > deadline:
                break
            if fa * fb > 0:
                break
            if abs(b - a) < 1e-15:
                break
            
            # Try IQUO if we have three points, else use secant/bisection
            xn = None
            if x_prev is not None and f_prev is not None and abs(f_prev) > 1e-15:
                # IQUO: interpolate x as function of f
                denom1 = (f_prev - fa)
                denom2 = (f_prev - fb)
                denom3 = (fb - fa)
                if abs(denom1) > 1e-15 and abs(denom2) > 1e-15 and abs(denom3) > 1e-15:
                    num = (fa * fb * (x_prev - b) / (f_prev - fa) * (f_prev - fb) / (fb - fa))
                    # Standard IQUO formula:
                    # x = (fa^2*(x_prev-xb)/(fa-fb)*(fa-fp) + fb^2*(x_prev-xa)/(fb-fa)*(fb-fp) + fp^2*(xa-xb)/(fp-fa)*(fp-fb))
                    #     / (fa*(fa-fb)*(fa-fp) + fb*(fb-fa)*(fb-fp) + fp*(fp-fa)*(fp-fb))
                    # Simpler: use secant as fallback
                    pass
            
            # Use secant (Newton-like) step from b
            if fb != fa and abs(b - a) > 1e-15:
                deriv = (fb - fa) / (b - a)
            else:
                deriv = 1.0
            
            if abs(deriv) > 1e-15:
                xn = b - fb / deriv
            else:
                xn = (a + b) / 2.0
            
            # Keep xn in bracket
            if not (a < xn < b):
                xn = (a + b) / 2.0
            
            fxn = safe_f(xn)
            
            if abs(fxn) <= TOL:
                # Do a polish pass with tighter tolerance
                x_curr = xn
                f_curr = fxn
                x_prev2 = b
                f_prev2 = fb
                for _ in range(10):
                    if calls >= MAX_CALLS - 5 or time.time() > deadline:
                        break
                    if abs(f_curr) <= TOL_POLISH:
                        break
                    if f_curr == f_prev2:
                        break
                    x_new2 = x_curr - f_curr * (x_curr - x_prev2) / (f_curr - f_prev2)
                    f_new2 = safe_f(x_new2)
                    if not math.isfinite(f_new2):
                        break
                    x_prev2, f_prev2 = x_curr, f_curr
                    x_curr, f_curr = x_new2, f_new2
                    if abs(f_curr) < abs(f_prev2):
                        best_x = x_curr
                        best_abs = abs(f_curr)
                return x_curr
            
            if not math.isfinite(fxn):
                xn = (a + b) / 2.0
                fxn = safe_f(xn)
                if not math.isfinite(fxn):
                    break
            
            # Update bracket
            if fa * fxn < 0:
                b = xn
                fb = fxn
            else:
                a = xn
                fa = fxn
        
        # Final polish pass on the best endpoint
        if abs(fa) < abs(fb):
            x_final = a
            f_final = fa
        else:
            x_final = b
            f_final = fb
        
        # Polish with secant using tighter tolerance
        if math.isfinite(f_final) and abs(f_final) < 1e-4:
            # Need another point for secant
            x_other = b if x_final == a else a
            f_other = fb if x_final == a else fa
            if math.isfinite(f_other) and f_final != f_other:
                for _ in range(8):
                    if calls >= MAX_CALLS - 5 or time.time() > deadline:
                        break
                    if abs(f_final) <= TOL_POLISH:
                        break
                    x_new = x_final - f_final * (x_final - x_other) / (f_final - f_other)
                    f_new = safe_f(x_new)
                    if not math.isfinite(f_new):
                        break
                    x_other, f_other = x_final, f_final
                    x_final, f_final = x_new, f_new
                    if abs(f_final) < best_abs:
                        best_x = x_final
                        best_abs = abs(f_final)
                return x_final
        
        return x_final
    
    # No bracket found - use secant method iteratively
    # This works well for smooth oscillatory functions
    x_prev = x0 - 0.1
    fx_prev = safe_f(x_prev)
    x_curr = x0
    fx_curr = safe_f(x_curr)
    
    if abs(fx_curr) <= TOL:
        return x_curr
    
    for _ in range(80):
        if calls >= MAX_CALLS - 10 or time.time() > deadline:
            break
        
        if abs(fx_prev) < abs(fx_curr):
            # Swap to keep convergence
            x_prev, x_curr = x_curr, x_prev
            fx_prev, fx_curr = fx_curr, fx_prev
        
        if abs(fx_curr) <= TOL:
            # Polish pass
            for _ in range(10):
                if calls >= MAX_CALLS - 5 or time.time() > deadline:
                    break
                if abs(fx_curr) <= TOL_POLISH:
                    break
                if fx_curr == fx_prev:
                    break
                x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                fx_new = safe_f(x_new)
                if not math.isfinite(fx_new):
                    break
                x_prev, fx_prev = x_curr, fx_curr
                x_curr, fx_curr = x_new, fx_new
                if abs(fx_curr) < best_abs:
                    best_x = x_curr
                    best_abs = abs(fx_curr)
            return x_curr
        
        if fx_prev == fx_curr:
            break
        
        x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        # Damp if step is too large
        if abs(x_new - x_curr) > 10.0 * max(1.0, abs(x_curr)):
            x_new = x_curr - 10.0 * max(1.0, abs(x_curr)) * (1.0 if fx_curr > 0 else -1.0)
        
        fx_new = safe_f(x_new)
        
        if abs(fx_new) <= TOL:
            return x_new
        
        if not math.isfinite(fx_new):
            # Try smaller step
            x_new = x_curr - 0.1 * (x_curr - x_prev)
            fx_new = safe_f(x_new)
            if not math.isfinite(fx_new):
                break
        
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_new, fx_new
    
    # Return best found
    return best_x
