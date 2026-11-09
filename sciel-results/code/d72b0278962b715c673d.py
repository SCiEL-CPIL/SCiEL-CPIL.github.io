import math

def find_root(f, x0):
    x = float(x0)
    try:
        fx = f(x)
    except Exception:
        return x

    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    # We'll do a local search for a sign change, then bisect.
    # Strategy: try small steps, then geometrically increasing steps in both directions.
    # We keep the best point (smallest |f|) as fallback.

    best_x = x
    best_fabs = abs(fx)

    # Initial step: relative to magnitude of x, but at least some absolute value
    # For oscillatory functions, we might need to find a nearby zero.
    # Use a mix of absolute and relative steps.
    base = max(1e-4, 1e-6 * max(1.0, abs(x)))
    
    # Try a sequence of steps: base, 2*base, 4*base, ... up to some max
    # In both + and - directions
    step = base
    max_step = 1e6  # Reasonable max step for most problems
    iterations = 0
    max_iter = 100  # Limit the search iterations to avoid infinite loops
    
    while step < max_step and iterations < max_iter:
        # Check x + step
        xp = x + step
        try:
            fxp = f(xp)
        except Exception:
            fxp = float('nan')
        
        if math.isfinite(fxp):
            if abs(fxp) <= 1e-8:
                return xp
            if abs(fxp) < best_fabs:
                best_fabs = abs(fxp)
                best_x = xp
            # Check for sign change with x
            if fx * fxp < 0:
                # Found a bracket: [x, xp]
                a, b = x, xp
                fa, fb = fx, fxp
                # Bisect to find root
                for _ in range(100):
                    m = 0.5 * (a + b)
                    fm = f(m)
                    if abs(fm) <= 1e-8:
                        return m
                    if not math.isfinite(fm):
                        break
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                return 0.5 * (a + b)
        
        # Check x - step
        xm = x - step
        try:
            fxm = f(xm)
        except Exception:
            fxm = float('nan')
        
        if math.isfinite(fxm):
            if abs(fxm) <= 1e-8:
                return xm
            if abs(fxm) < best_fabs:
                best_fabs = abs(fxm)
                best_x = xm
            # Check for sign change with x
            if fx * fxm < 0:
                # Found a bracket: [xm, x]
                a, b = xm, x
                fa, fb = fxm, fx
                # Bisect to find root
                for _ in range(100):
                    m = 0.5 * (a + b)
                    fm = f(m)
                    if abs(fm) <= 1e-8:
                        return m
                    if not math.isfinite(fm):
                        break
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                return 0.5 * (a + b)
        
        step *= 2.0
        iterations += 1
    
    # If we didn't find a bracket, try a more thorough search with smaller steps
    # This helps for functions with narrow roots or oscillatory behavior
    # Try a grid search in a local region
    local_radius = max(0.1, 0.01 * max(1.0, abs(x)))
    n_grid = 20
    dx = 2 * local_radius / n_grid
    
    for i in range(n_grid + 1):
        xi = x - local_radius + i * dx
        try:
            fxi = f(xi)
        except Exception:
            continue
        
        if not math.isfinite(fxi):
            continue
            
        if abs(fxi) <= 1e-8:
            return xi
        if abs(fxi) < best_fabs:
            best_fabs = abs(fxi)
            best_x = xi
            
        # Check for sign change with previous grid point
        if i > 0:
            x_prev = x - local_radius + (i - 1) * dx
            try:
                f_prev = f(x_prev)
            except Exception:
                continue
            if math.isfinite(f_prev) and f_prev * fxi < 0:
                a, b = x_prev, xi
                fa, fb = f_prev, fxi
                for _ in range(80):
                    m = 0.5 * (a + b)
                    fm = f(m)
                    if abs(fm) <= 1e-8:
                        return m
                    if not math.isfinite(fm):
                        break
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                return 0.5 * (a + b)
    
    # As a last resort, try scipy's brentq with a wider bracket if possible
    # First, try to find a bracket by scanning
    a = None
    b = None
    fa = None
    fb = None
    
    # Scan over a range of points to find a sign change
    scan_range = max(1.0, 10.0 * max(1.0, abs(x)))
    n_scan = 50
    dx_scan = 2 * scan_range / n_scan
    
    prev_x = x - scan_range
    try:
        prev_fx = f(prev_x)
    except Exception:
        prev_fx = float('nan')
    
    for i in range(1, n_scan + 1):
        xi = x - scan_range + i * dx_scan
        try:
            fxi = f(xi)
        except Exception:
            fxi = float('nan')
        
        if math.isfinite(fxi) and math.isfinite(prev_fx):
            if prev_fx * fxi < 0:
                a = prev_x
                b = xi
                fa = prev_fx
                fb = fxi
                break
            if abs(fxi) <= 1e-8:
                return xi
            if abs(fxi) < best_fabs:
                best_fabs = abs(fxi)
                best_x = xi
        else:
            if math.isfinite(fxi) and abs(fxi) < best_fabs:
                best_fabs = abs(fxi)
                best_x = xi
        
        if math.isfinite(fxi):
            prev_x = xi
            prev_fx = fxi
    
    if a is not None:
        try:
            from scipy.optimize import brentq
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            pass
    
    return best_x
