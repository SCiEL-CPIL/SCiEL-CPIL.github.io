"""Root finder for 3 fixed oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using adaptive geometric expansion for bracketing
    and Brent's method with Inverse Quadratic Interpolation (IQI) for root refinement.
    """
    tol = 1e-8
    deadline = time.time() + 110.0  # safety margin before 120s wall-clock limit
    
    x = float(x0)
    
    # Evaluate at starting point
    try:
        fx = f(x)
    except Exception:
        fx = float('nan')
    
    if not math.isfinite(fx):
        # Try a few nearby points
        for dx in [1e-6, -1e-6, 1e-4, -1e-4, 1.0, -1.0]:
            try:
                x_try = x + dx
                fx_try = f(x_try)
                if math.isfinite(fx_try) and abs(fx_try) <= tol:
                    return x_try
            except Exception:
                continue
        return x  # best effort
    
    if abs(fx) <= tol:
        return x
    
    # Phase 1: Find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x
    fa = fx
    
    # Adaptive geometric expansion: try both directions with exponentially growing steps
    # This reduces calls compared to fixed-step grid scan
    b = None
    fb = None
    
    # Try positive direction
    step = 1e-8
    while step <= 1e6:
        if time.time() > deadline:
            break
        x_try = x + step
        try:
            f_try = f(x_try)
        except Exception:
            step *= 10
            continue
        
        if not math.isfinite(f_try):
            step *= 10
            continue
        
        if abs(f_try) <= tol:
            return x_try
        
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break
        step *= 10
    
    # If no bracket found in positive direction, try negative direction
    if b is None:
        step = 1e-8
        while step <= 1e6:
            if time.time() > deadline:
                break
            x_try = x - step
            try:
                f_try = f(x_try)
            except Exception:
                step *= 10
                continue
            
            if not math.isfinite(f_try):
                step *= 10
                continue
            
            if abs(f_try) <= tol:
                return x_try
            
            if fa * f_try < 0:
                b = x_try
                fb = f_try
                break
            step *= 10
    
    if b is None:
        # No bracket found with geometric expansion.
        # Try a more systematic search: sample on a grid around x0
        best_x = x
        best_abs_fx = abs(fx)
        
        # Sample on a logarithmic grid
        for scale in [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]:
            if time.time() > deadline:
                break
            for sign in [1, -1]:
                x_try = x + sign * scale
                try:
                    f_try = f(x_try)
                except Exception:
                    continue
                if not math.isfinite(f_try):
                    continue
                if abs(f_try) < best_abs_fx:
                    best_abs_fx = abs(f_try)
                    best_x = x_try
                if abs(f_try) <= tol:
                    return x_try
        
        # Try to bracket using the best point found
        if best_abs_fx < abs(fx):
            a = best_x
            fa = f(best_x)
            if abs(fa) <= tol:
                return a
        
        # If still no bracket, try secant method as fallback
        x1 = best_x
        f1 = fa if abs(fa) < abs(fx) else fx
        x0_sec = x
        f0_sec = fx
        
        if x1 != x0_sec and f1 != f0_sec:
            for _ in range(100):
                if time.time() > deadline:
                    break
                if f1 == f0_sec:
                    break
                x_new = x1 - f1 * (x1 - x0_sec) / (f1 - f0_sec)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break
                if abs(f_new) <= tol:
                    return x_new
                x0_sec = x1
                f0_sec = f1
                x1 = x_new
                f1 = f_new
                if abs(x1 - x0_sec) < 1e-15 * max(1.0, abs(x1)):
                    break
        
        return best_x
    
    # Phase 2: We have a bracket [a, b] with f(a)*f(b) < 0
    # Use Brent's method with Inverse Quadratic Interpolation (IQI)
    
    # Initialize Brent's method state
    # c is the previous x, d is the current x
    # We need three points for IQI: a, c, b (or a, b, c depending on which is most recent)
    
    # Start with a and b as our initial points
    # c will be the most recent point, d will be the second most recent
    # For the first iteration, we only have a and b, so we use bisection
    
    x_a = a
    x_b = b
    f_a = fa
    f_b = fb
    
    # Track the last few points for IQI
    # We'll maintain a list of recent (x, f(x)) pairs
    recent_points = [(x_a, f_a), (x_b, f_b)]
    
    max_iter = 200
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check if we've already found a root
        if abs(f_a) <= tol:
            return x_a
        if abs(f_b) <= tol:
            return x_b
        
        # Try IQI if we have at least 3 distinct points with distinct function values
        x_new = None
        use_iqi = False
        
        if len(recent_points) >= 3:
            # Get the three most recent points
            p1, p2, p3 = recent_points[-3], recent_points[-2], recent_points[-1]
            x1, f1 = p1
            x2, f2 = p2
            x3, f3 = p3
            
            # Check for distinct x values and function values
            if (x1 != x2 and x2 != x3 and x1 != x3 and 
                f1 != f2 and f2 != f3 and f1 != f3):
                
                # Inverse Quadratic Interpolation
                # Find x such that f(x) = 0 using quadratic interpolation
                # f(x) ≈ A(x-x1)(x-x2) + B(x-x3) + C
                # Actually, IQI interpolates x as a function of f
                # x(f) ≈ quadratic in f passing through (f1,x1), (f2,x2), (f3,x3)
                
                # Using the standard IQI formula:
                # x = (f1*f2*(x3 - x1) + f1*f3*(x1 - x2) + f2*f3*(x2 - x3)) / 
                #     (f1*(f2 - f3) + f2*(f3 - f1) + f3*(f1 - f2))
                
                denom = (f1 * (f2 - f3) + f2 * (f3 - f1) + f3 * (f1 - f2))
                
                if abs(denom) > 1e-30:
                    numer = (f1 * f2 * (x3 - x1) + f1 * f3 * (x1 - x2) + f2 * f3 * (x2 - x3))
                    x_iqi = numer / denom
                    
                    # Check if IQI point is within the bracket
                    if (min(x_a, x_b) < x_iqi < max(x_a, x_b)) and math.isfinite(x_iqi):
                        x_new = x_iqi
                        use_iqi = True
        
        if not use_iqi:
            # Fall back to bisection
            x_new = (x_a + x_b) / 2.0
        
        # Evaluate f at the new point
        try:
            f_new = f(x_new)
        except Exception:
            # If evaluation fails, use bisection
            x_new = (x_a + x_b) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if not math.isfinite(f_new):
            x_new = (x_a + x_b) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if abs(f_new) <= tol:
            return x_new
        
        # Update bracket
        if f_a * f_new < 0:
            x_b = x_new
            f_b = f_new
        else:
            x_a = x_new
            f_a = f_new
        
        # Update recent points (keep last 3)
        recent_points.append((x_new, f_new))
        if len(recent_points) > 3:
            recent_points.pop(0)
        
        # Check convergence
        if abs(x_b - x_a) < 1e-15 * max(1.0, abs(x_new)):
            break
    
    # Return the point with smallest |f|
    candidates = [x_a, x_b]
    best_x = x_a if abs(f_a) < abs(f_b) else x_b
    best_val = min(abs(f_a), abs(f_b))
    
    # Also check recent points
    for px, pf in recent_points:
        if abs(pf) < best_val and math.isfinite(pf):
            best_val = abs(pf)
            best_x = px
    
    return best_x

# EVOLVE-BLOCK-END
