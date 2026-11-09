"""Root finder for 3 fixed oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using an adaptive cached grid scan
    followed by high-precision bisection/secant hybrid refinement.
    """
    tol = 1e-8
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    x = float(x0)
    
    # Cache to avoid redundant function calls
    cache = {}
    
    def eval_f(xv):
        """Evaluate f with caching."""
        xv = float(xv)
        if xv in cache:
            return cache[xv]
        try:
            val = f(xv)
        except Exception:
            val = float('nan')
        cache[xv] = val
        return val
    
    # Track best candidate
    best_x = x
    best_abs = float('inf')
    
    def update_best(xv, fv):
        nonlocal best_x, best_abs
        if math.isfinite(fv) and abs(fv) < best_abs:
            best_abs = abs(fv)
            best_x = xv
    
    # Evaluate at starting point
    fx = eval_f(x)
    update_best(x, fx)
    
    if math.isfinite(fx) and abs(fx) <= tol:
        return x
    
    # Phase 1: Adaptive grid scan to find a bracket
    # Start with a small step and expand geometrically
    # The cache ensures we don't re-evaluate points
    
    a = x
    fa = fx
    
    if not math.isfinite(fa):
        # If f(x0) is not finite, try nearby points
        for dx in [1e-6, -1e-6, 1e-4, -1e-4, 1.0, -1.0, 10.0, -10.0]:
            if time.time() > deadline:
                break
            x_try = x + dx
            fx_try = eval_f(x_try)
            update_best(x_try, fx_try)
            if math.isfinite(fx_try) and abs(fx_try) <= tol:
                return x_try
        return best_x
    
    # Geometric expansion to find a bracket
    # Try both directions with increasing step sizes
    step = 1e-6
    max_steps = 60  # step grows to ~1e-6 * 2^60 which is huge
    found_bracket = False
    b = None
    fb = None
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        
        # Try positive direction
        x_try = x + step
        fx_try = eval_f(x_try)
        update_best(x_try, fx_try)
        
        if math.isfinite(fx_try):
            if abs(fx_try) <= tol:
                return x_try
            if fa * fx_try < 0:
                b = x_try
                fb = fx_try
                found_bracket = True
                break
        
        # Try negative direction
        x_try = x - step
        fx_try = eval_f(x_try)
        update_best(x_try, fx_try)
        
        if math.isfinite(fx_try):
            if abs(fx_try) <= tol:
                return x_try
            if fa * fx_try < 0:
                b = x_try
                fb = fx_try
                found_bracket = True
                break
        
        step *= 2.0
    
    # If no bracket found with geometric expansion, try a denser scan
    # around the best point found so far
    if not found_bracket:
        # Scan around best_x with small steps
        scan_step = 1e-4
        for offset in range(-200, 201):
            if time.time() > deadline:
                break
            x_try = best_x + offset * scan_step
            fx_try = eval_f(x_try)
            update_best(x_try, fx_try)
            if math.isfinite(fx_try) and abs(fx_try) <= tol:
                return x_try
            if offset > 0:
                x_prev = best_x + (offset - 1) * scan_step
                fx_prev = eval_f(x_prev)
                if math.isfinite(fx_prev) and fx_prev * fx_try < 0:
                    a = x_prev
                    fa = fx_prev
                    b = x_try
                    fb = fx_try
                    found_bracket = True
                    break
    
    if not found_bracket:
        # Last resort: secant method from best two points
        # Find two distinct points with finite f values
        points = [(k, v) for k, v in cache.items() if math.isfinite(v)]
        points.sort(key=lambda kv: abs(kv[1]))
        
        if len(points) >= 2:
            x1, f1 = points[0]
            x2, f2 = points[1]
            
            if x1 != x2 and f1 != f2:
                for _ in range(80):
                    if time.time() > deadline:
                        break
                    x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                    f_new = eval_f(x_new)
                    update_best(x_new, f_new)
                    if math.isfinite(f_new) and abs(f_new) <= tol:
                        return x_new
                    if not math.isfinite(f_new):
                        break
                    x2, f2 = x1, f1
                    x1, f1 = x_new, f_new
        
        return best_x
    
    # Phase 2: High-precision refinement within bracket [a, b]
    # Use Illinois method (modified bisection) for robustness
    x_lo = a
    x_hi = b
    f_lo = fa
    f_hi = fb
    
    # Ensure f_lo and f_hi have opposite signs
    if f_lo * f_hi > 0:
        # Swap if needed
        x_lo, x_hi = x_hi, x_lo
        f_lo, f_hi = f_hi, f_lo
    
    for _ in range(80):
        if time.time() > deadline:
            break
        
        # Illinois method: use weighted midpoint
        if abs(f_lo) < abs(f_hi):
            # f_lo is smaller, move hi toward lo
            x_mid = (x_lo + x_hi) / 2.0
            f_mid = eval_f(x_mid)
            update_best(x_mid, f_mid)
            
            if not math.isfinite(f_mid):
                break
            
            if abs(f_mid) <= tol:
                return x_mid
            
            if f_lo * f_mid < 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                # Keep lo, but weight hi
                x_lo = x_hi
                f_lo = f_hi
                x_hi = x_mid
                f_hi = f_mid
                # Illinois: scale the side we're keeping
                f_lo *= 0.5
        else:
            x_mid = (x_lo + x_hi) / 2.0
            f_mid = eval_f(x_mid)
            update_best(x_mid, f_mid)
            
            if not math.isfinite(f_mid):
                break
            
            if abs(f_mid) <= tol:
                return x_mid
            
            if f_lo * f_mid < 0:
                x_hi = x_mid
                f_hi = f_mid
                f_hi *= 0.5
            else:
                x_lo = x_mid
                f_lo = f_mid
        
        # Check convergence
        if abs(x_hi - x_lo) < 1e-15 * max(1.0, abs((x_lo + x_hi) / 2.0)):
            break
    
    # Final check: return the best candidate
    if best_abs <= tol:
        return best_x
    
    # Try one more polish: secant from the two bracket endpoints
    if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo != f_hi:
        x_sec = x_lo - f_lo * (x_lo - x_hi) / (f_lo - f_hi)
        f_sec = eval_f(x_sec)
        update_best(x_sec, f_sec)
        if math.isfinite(f_sec) and abs(f_sec) <= tol:
            return x_sec
    
    return best_x

# EVOLVE-BLOCK-END
