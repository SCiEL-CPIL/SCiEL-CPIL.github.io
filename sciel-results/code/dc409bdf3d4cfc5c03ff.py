import time

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_f = None
    
    def call_f(x):
        nonlocal evals, best_x, best_f
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = float(f(x))
        except:
            return None
        if best_f is None or abs(val) < abs(best_f):
            best_f = val
            best_x = x
        return val

    # 1. Initial evaluation
    fx = call_f(float(x0))
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return float(x0)
        
    # 2. Try to find a bracket
    a, b = None, None
    fa, fb = None, None
    x_curr = float(x0)
    
    # Search for bracket by expanding
    step = 1e-3
    found_bracket = False
    # Try right
    for i in range(20):
        x_try = x_curr + step
        f_try = call_f(x_try)
        if f_try is None: break
        if fx * f_try < 0:
            a, b = x_curr, x_try
            fa, fb = fx, f_try
            found_bracket = True
            break
        # Update current if it's better? No, we are looking for sign change.
        # But if we go too far, we might miss a local root. 
        # For "local" ill-conditioned, usually the root is near x0.
        step *= 2.0
        
    if not found_bracket:
        # Try left
        step = 1e-3
        for i in range(20):
            x_try = x_curr - step
            f_try = call_f(x_try)
            if f_try is None: break
            if fx * f_try < 0:
                a, b = x_try, x_curr
                fa, fb = f_try, fx
                found_bracket = True
                break
            step *= 2.0

    # 3. Solve
    if found_bracket:
        # Bisection/Secant Hybrid on [a, b]
        # Ensure fa < fb in magnitude? No, just keep track.
        # We want |f(x)| small.
        while evals < MAX_EVALS and time.monotonic() < DEADLINE:
            # Check convergence
            if abs(fa) <= TOL: return a
            if abs(fb) <= TOL: return b
            
            # Try Secant
            # x_sec = b - fb * (b - a) / (fb - fa)
            denom = fb - fa
            if abs(denom) < 1e-15:
                # Fall back to bisection
                x_mid = (a + b) / 2.0
            else:
                x_mid = b - fb * (b - a) / denom
            
            # If Secant step is outside or too aggressive, use Bisection
            # A good heuristic: if x_mid is not in [a, b], use bisection.
            # Or if it doesn't reduce the interval by a factor, use bisection.
            if not (a < x_mid < b):
                x_mid = (a + b) / 2.0
            
            f_mid = call_f(x_mid)
            if f_mid is None: break
            
            if abs(f_mid) <= TOL:
                return x_mid
            
            # Update bracket
            if fa * f_mid <= 0:
                b = x_mid
                fb = f_mid
            else:
                a = x_mid
                fa = f_mid
    else:
        # No bracket found. Use Newton with damping.
        x = float(x0)
        fx = call_f(x)
        if fx is None: return best_x
        if abs(fx) <= TOL: return x
        
        for _ in range(MAX_EVALS - evals):
            if time.monotonic() > DEADLINE: break
            
            # Estimate derivative
            h = 1e-6 * max(1.0, abs(x))
            fx_h = call_f(x + h)
            if fx_h is None: break
            df = (fx_h - fx) / h
            
            if df == 0:
                break
            
            dx = fx / df
            x_new = x - dx
            
            # Damping: if step is too big, shrink it
            # "Too big" is subjective. Let's say if |dx| > 10 * max(1, |x|), shrink.
            # Or simply check if f(x_new) is better.
            
            f_new = call_f(x_new)
            if f_new is None: break
            
            if abs(f_new) <= TOL:
                return x_new
            
            if abs(f_new) < abs(fx):
                x = x_new
                fx = f_new
            else:
                # Step was bad. Try half step.
                x_new = x - 0.5 * dx
                f_new = call_f(x_new)
                if f_new is None: break
                if abs(f_new) < abs(fx):
                    x = x_new
                    fx = f_new
                else:
                    # Try even smaller or break
                    # If we keep failing, maybe we are stuck in a flat region.
                    # Break to avoid wasting evals.
                    break
            if abs(fx) <= TOL:
                return x

    return best_x
