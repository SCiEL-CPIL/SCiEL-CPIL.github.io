"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder using Safeguarded Secant/Bisection (SIQI-style) with
    a golden-section minimization fallback for non-sign-changing (tangential) roots.
    """
    deadline = time.time() + 142  # Leave 8s margin before 150s kill
    max_evals = 495
    eval_count = [0]
    best_x = float(x0)
    best_fx = None
    best_abs_fx = float('inf')
    
    def eval_f(x):
        """Evaluate f with counting and deadline check."""
        nonlocal best_x, best_fx, best_abs_fx
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if val is None or math.isnan(val) or math.isinf(val):
                return None
            abs_val = abs(val)
            if abs_val < best_abs_fx:
                best_abs_fx = abs_val
                best_x = x
                best_fx = val
            return val
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError, TypeError):
            return None
    
    def is_good(x, fx):
        return fx is not None and not math.isnan(fx) and not math.isinf(fx) and abs(fx) <= 1e-8
    
    def time_ok():
        return time.time() < deadline
    
    def evals_ok():
        return eval_count[0] < max_evals
    
    # Phase 1: Quick check at x0
    fx0 = eval_f(float(x0))
    if fx0 is not None and is_good(x0, fx0):
        return x0
    
    # Phase 2: Find a bracket or collect points
    x0_f = float(x0)
    # Try to find a sign change by expanding in both directions
    bracket_a = None
    bracket_b = None
    fa = None
    fb = None
    
    if fx0 is not None and not math.isnan(fx0) and not math.isinf(fx0):
        fa = fx0
        bracket_a = x0_f
        
        # Try expanding to find sign change
        h = 1e-3 * max(1.0, abs(x0_f))
        for _ in range(40):
            if not time_ok() or not evals_ok():
                break
            # Try positive direction
            x_try = bracket_a + h
            fx_try = eval_f(x_try)
            if fx_try is not None and not math.isnan(fx_try) and not math.isinf(fx_try):
                if is_good(x_try, fx_try):
                    return x_try
                if fa * fx_try < 0:
                    bracket_b = x_try
                    fb = fx_try
                    break
                # Keep the point with smaller |f| as new reference if it's closer to zero
                if abs(fx_try) < abs(fa):
                    # Shift bracket
                    pass
            h *= 2.0
            if h > 1e6 * max(1.0, abs(x0_f)):
                break
        
        # If no bracket found in positive direction, try negative
        if bracket_b is None:
            h = 1e-3 * max(1.0, abs(x0_f))
            for _ in range(40):
                if not time_ok() or not evals_ok():
                    break
                x_try = bracket_a - h
                fx_try = eval_f(x_try)
                if fx_try is not None and not math.isnan(fx_try) and not math.isinf(fx_try):
                    if is_good(x_try, fx_try):
                        return x_try
                    if fa * fx_try < 0:
                        bracket_b = x_try
                        fb = fx_try
                        break
                h *= 2.0
                if h > 1e6 * max(1.0, abs(x0_f)):
                    break
    
    # If we have a bracket, use safeguarded secant/bisection
    if bracket_a is not None and bracket_b is not None and fa is not None and fb is not None:
        a, b = bracket_a, bracket_b
        fa_val, fb_val = fa, fb
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa_val, fb_val = fb_val, fa_val
        
        # Safeguarded secant (SIQI-like) iteration
        for _ in range(200):
            if not time_ok() or not evals_ok():
                break
            
            # Check if endpoints are good
            if abs(fa_val) <= 1e-8:
                return a
            if abs(fb_val) <= 1e-8:
                return b
            
            # Secant step
            if fa_val != fb_val:
                x_sec = a - fa_val * (b - a) / (fb_val - fa_val)
            else:
                x_sec = (a + b) / 2
            
            # Safeguard: keep within bracket
            if x_sec <= a or x_sec >= b:
                x_sec = (a + b) / 2
            
            # Additional safeguard: don't take too small a step
            min_step = 1e-15 * max(1.0, abs(a), abs(b))
            if abs(x_sec - a) < min_step or abs(x_sec - b) < min_step:
                x_sec = (a + b) / 2
            
            fx_sec = eval_f(x_sec)
            if fx_sec is None:
                break
            if is_good(x_sec, fx_sec):
                return x_sec
            
            # Update bracket
            if fa_val * fx_sec <= 0:
                b = x_sec
                fb_val = fx_sec
            else:
                a = x_sec
                fa_val = fx_sec
        
        return best_x
    
    # No bracket found: use golden-section minimization on f(x)^2
    # to find a tangential root or the minimum of |f|
    x_min = x0_f
    f_min = abs(fx0) if fx0 is not None else float('inf')
    
    if f_min <= 1e-8:
        return x0_f
    
    # Golden section search for minimum of |f(x)|
    # Start with a reasonable interval
    h = 1e-2 * max(1.0, abs(x0_f))
    ga = x0_f - h
    gb = x0_f + h
    
    # First evaluate endpoints
    fga = eval_f(ga)
    fgb = eval_f(gb)
    
    if fga is not None and abs(fga) <= 1e-8:
        return ga
    if fgb is not None and abs(fgb) <= 1e-8:
        return gb
    
    # If we found a sign change during bracket search, we'd have caught it above
    # Now do golden section on g(x) = f(x)^2 (or |f(x)|)
    # Use the three-point set for better initial bracket
    if fga is not None and fgb is not None:
        if abs(fga) < f_min:
            x_min = ga
            f_min = abs(fga)
        if abs(fgb) < f_min:
            x_min = gb
            f_min = abs(fgb)
    
    # Expand interval if needed to ensure we're around the minimum
    for _ in range(10):
        if not time_ok() or not evals_ok():
            break
        if f_min <= 1e-8:
            return x_min
        h *= 2.0
        ga = x0_f - h
        gb = x0_f + h
        fga = eval_f(ga)
        fgb = eval_f(gb)
        if fga is not None and abs(fga) <= 1e-8:
            return ga
        if fgb is not None and abs(fgb) <= 1e-8:
            return gb
        if fga is not None and abs(fga) < f_min:
            x_min = ga
            f_min = abs(fga)
        if fgb is not None and abs(fgb) < f_min:
            x_min = gb
            f_min = abs(fgb)
        # Stop expanding if function values are increasing
        if fga is not None and fgb is not None:
            if abs(fga) > f_min * 10 and abs(fgb) > f_min * 10:
                break
    
    # Golden section search
    if fga is None or fgb is None:
        # Try smaller steps
        h = 1e-4 * max(1.0, abs(x0_f))
        for direction in [1, -1]:
            x_try = x0_f + direction * h
            fx_try = eval_f(x_try)
            if fx_try is not None and abs(fx_try) <= 1e-8:
                return x_try
            if fx_try is not None and abs(fx_try) < f_min:
                x_min = x_try
                f_min = abs(fx_try)
    
    # Now do golden section on the interval [ga, gb]
    # We want to minimize |f(x)|
    phi = (math.sqrt(5) - 1) / 2  # ~0.618
    inv_phi = 1.0 - phi  # ~0.382
    
    # Set up golden section
    if fga is not None and fgb is not None:
        a_gs = ga
        b_gs = gb
        fa_gs = abs(fga)
        fb_gs = abs(fgb)
        
        # Interior points
        c_gs = a_gs + inv_phi * (b_gs - a_gs)
        d_gs = a_gs + phi * (b_gs - a_gs)
        
        fc_gs = eval_f(c_gs)
        fd_gs = eval_f(d_gs)
        
        if fc_gs is not None and abs(fc_gs) <= 1e-8:
            return c_gs
        if fd_gs is not None and abs(fd_gs) <= 1e-8:
            return d_gs
        
        for _ in range(100):
            if not time_ok() or not evals_ok():
                break
            
            if fc_gs is not None and abs(fc_gs) < f_min:
                f_min = abs(fc_gs)
                x_min = c_gs
            if fd_gs is not None and abs(fd_gs) < f_min:
                f_min = abs(fd_gs)
                x_min = d_gs
            
            if f_min <= 1e-8:
                return x_min
            
            if fc_gs is None or fd_gs is None:
                break
            
            if fc_gs < fd_gs:
                b_gs = d_gs
                fb_gs = fd_gs
                d_gs = c_gs
                fd_gs = fc_gs
                c_gs = a_gs + inv_phi * (b_gs - a_gs)
                fc_gs = eval_f(c_gs)
            else:
                a_gs = c_gs
                fa_gs = fc_gs
                c_gs = d_gs
                fc_gs = fd_gs
                d_gs = a_gs + phi * (b_gs - a_gs)
                fd_gs = eval_f(d_gs)
    
    # Final polish: if we're close, do secant iterations
    if best_abs_fx < 1e-4 and best_fx is not None:
        x_polish = best_x
        fx_polish = best_fx
        h_polish = 1e-8 * max(1.0, abs(x_polish))
        x_prev = x_polish + h_polish
        fx_prev = eval_f(x_prev)
        
        for _ in range(30):
            if not time_ok() or not evals_ok():
                break
            if fx_prev is None or math.isnan(fx_prev) or math.isinf(fx_prev):
                break
            if fx_polish == fx_prev:
                break
            denom = fx_polish - fx_prev
            if abs(denom) < 1e-30:
                break
            x_new = x_polish - fx_polish * (x_polish - x_prev) / denom
            if math.isnan(x_new) or math.isinf(x_new):
                break
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            if is_good(x_new, fx_new):
                return x_new
            x_prev, fx_prev = x_polish, fx_polish
            x_polish, fx_polish = x_new, fx_new
    
    return best_x
# EVOLVE-BLOCK-END
