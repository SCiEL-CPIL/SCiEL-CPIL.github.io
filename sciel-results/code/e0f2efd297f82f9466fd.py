import math
import time

def find_root(f, x0):
    tol = 1e-8
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= tol:
        return x
    
    # Track best point found
    best_x = x
    best_f = abs(fx)
    
    # Try to find a bracket [a, b] where f(a) * f(b) < 0
    # Start with small steps, then expand geometrically
    a = x
    fa = fx
    
    # First check nearby points for a sign change
    deltas = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    bracket_found = False
    b = None
    fb = None
    
    for delta in deltas:
        if time.time() > deadline:
            return best_x  # Return best so far
        
        # Check x + delta
        x_plus = x + delta
        f_plus = f(x_plus)
        if abs(f_plus) <= tol:
            return x_plus
        if abs(f_plus) < best_f:
            best_x = x_plus
            best_f = abs(f_plus)
        if fa * f_plus < 0:
            b = x_plus
            fb = f_plus
            bracket_found = True
            break
        
        # Check x - delta
        x_minus = x - delta
        f_minus = f(x_minus)
        if abs(f_minus) <= tol:
            return x_minus
        if abs(f_minus) < best_f:
            best_x = x_minus
            best_f = abs(f_minus)
        if fa * f_minus < 0:
            b = x_minus
            fb = f_minus
            bracket_found = True
            break
    
    # If no bracket found with initial deltas, try larger geometric expansion
    if not bracket_found:
        step = 1.0
        for i in range(50):
            if time.time() > deadline:
                return best_x
            
            # Try expanding in both directions
            for sign in [1, -1]:
                x_try = x + sign * step
                f_try = f(x_try)
                if abs(f_try) <= tol:
                    return x_try
                if abs(f_try) < best_f:
                    best_x = x_try
                    best_f = abs(f_try)
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    bracket_found = True
                    break
            if bracket_found:
                break
            step *= 2.0
    
    if not bracket_found:
        # No bracket found; return the point with smallest |f|
        return best_x
    
    # Hybrid Secant/Bisection on [a, b]
    lo = a
    hi = b
    flo = fa
    fhi = fb
    
    # Keep track of last two points for secant method
    x_prev = None
    f_prev = None
    
    # Hybrid: try secant first, fall back to bisection if secant goes out of bracket
    for i in range(80):
        if time.time() > deadline:
            # Return midpoint of current bracket
            return (lo + hi) / 2.0
        
        # Try secant method if we have two points
        if x_prev is not None and f_prev is not None:
            denom = fhi - f_prev
            if abs(denom) > 1e-15:
                x_secant = hi - fhi * (hi - x_prev) / denom
                # Check if secant point is within bracket
                if lo <= x_secant <= hi:
                    f_secant = f(x_secant)
                    if abs(f_secant) <= tol:
                        return x_secant
                    if abs(f_secant) < best_f:
                        best_x = x_secant
                        best_f = abs(f_secant)
                    
                    # Update bracket based on sign change
                    if flo * f_secant < 0:
                        hi = x_secant
                        fhi = f_secant
                    else:
                        lo = x_secant
                        flo = f_secant
                    
                    # Update secant history
                    x_prev = hi
                    f_prev = fhi
                    continue
        
        # Fall back to bisection
        mid = (lo + hi) / 2.0
        fmid = f(mid)
        
        if abs(fmid) <= tol:
            return mid
        if abs(fmid) < best_f:
            best_x = mid
            best_f = abs(fmid)
        
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
        
        # Update secant history for next iteration
        x_prev = mid
        f_prev = fmid
        
        # If bracket is small enough, we're done
        if abs(hi - lo) < 1e-12:
            break
    
    # Final check: return best point if it satisfies tolerance
    if abs(f(best_x)) <= tol:
        return best_x
    
    return (lo + hi) / 2.0
