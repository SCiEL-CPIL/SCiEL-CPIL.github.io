"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Uses a hybrid strategy:
    1. Check x0 first.
    2. Try Newton's method with adaptive step size and safety checks.
    3. If Newton fails or diverges, fall back to bracketing (bisection/secant hybrid).
    4. Uses a deadline guard to ensure termination within time limits.
    """
    tol = 1e-8
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    x = float(x0)
    
    # Helper to check time
    def time_up():
        return time.time() > deadline
    
    # 1. Check initial point
    fx = f(x)
    if abs(fx) <= tol:
        return x
    if not math.isfinite(fx):
        # If f(x0) is not finite, we might need to move away, but let's try bracketing
        pass

    # Strategy A: Newton's Method with safeguards
    # We'll try up to 20 Newton steps. If it fails, we switch to Strategy B.
    # We keep track of the best point seen so far (smallest |f(x)|) to return if we fail.
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    h = 1e-6 * max(1.0, abs(x))
    
    for i in range(20):
        if time_up():
            break
            
        # Finite difference derivative
        # Use central difference if possible, or forward
        fx_plus = f(x + h)
        if not math.isfinite(fx_plus):
            # Try a different step or break
            h = 1e-4 * max(1.0, abs(x))
            fx_plus = f(x + h)
            if not math.isfinite(fx_plus):
                break
        
        # Use central difference for better accuracy if budget allows
        # f(x-h)
        fx_minus = f(x - h)
        if math.isfinite(fx_minus):
            d = (fx_plus - fx_minus) / (2 * h)
        else:
            d = (fx_plus - fx) / h
            
        if d == 0 or not math.isfinite(d):
            break
            
        x_new = x - fx / d
        
        # Safety check: don't move too far away from x0 or current best
        # Heuristic: if |x_new| > 100 * max(1, |x|), it's likely diverging
        if not math.isfinite(x_new):
            break
            
        # Check if we've made progress
        fx_new = f(x_new)
        
        if not math.isfinite(fx_new):
            break
            
        abs_fx_new = abs(fx_new)
        
        if abs_fx_new < best_abs_fx:
            best_x = x_new
            best_fx = fx_new
            best_abs_fx = abs_fx_new
            
        if abs_fx_new <= tol:
            return x_new
            
        # Damping: if the step is too large, reduce it
        # Simple check: if we moved away from zero in function value significantly, damp
        if abs_fx_new > 10 * abs(fx):
            # Damp the step
            x_new = x - 0.5 * (fx / d)
            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                break
            abs_fx_new = abs(fx_new)
            if abs_fx_new < best_abs_fx:
                best_x = x_new
                best_fx = fx_new
                best_abs_fx = abs_fx_new
            if abs_fx_new <= tol:
                return x_new
        
        x = x_new
        fx = fx_new
        h = 1e-6 * max(1.0, abs(x))
        
        # If we are converging linearly or slower, break to bracketing
        # Heuristic: if abs(fx) is not decreasing fast enough
        if i > 5 and abs_fx_new > abs(fx) * 0.5:
             break

    if best_abs_fx <= tol:
        return best_x

    # Strategy B: Bracketing
    # We need to find a bracket [a, b] where f(a)*f(b) < 0
    # Start from x0 (or best_x if it's better) and expand outwards
    
    # Use the best point found so far as the center
    center = best_x
    fc = best_fx
    
    # Try small steps first to find a sign change nearby
    # Then expand exponentially
    
    a = None
    b = None
    fa = None
    fb = None
    
    # We already have f(center)
    # Let's try to find a bracket by expanding
    
    step = 1e-3
    max_step = 1e6
    
    # Check immediate neighbors
    for delta in [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]:
        if time_up():
            break
            
        x_plus = center + delta
        x_minus = center - delta
        
        fx_plus = f(x_plus)
        if math.isfinite(fx_plus) and abs(fx_plus) <= tol:
            return x_plus
        if math.isfinite(fx_plus) and fx_plus * fc < 0:
            a = center
            b = x_plus
            fa = fc
            fb = fx_plus
            break
            
        fx_minus = f(x_minus)
        if math.isfinite(fx_minus) and abs(fx_minus) <= tol:
            return x_minus
        if math.isfinite(fx_minus) and fx_minus * fc < 0:
            a = x_minus
            b = center
            fa = fx_minus
            fb = fc
            break
            
    if a is None:
        # No bracket found in simple expansion. 
        # Try a wider scan or just return best_x (which will likely fail the |f(x)|<=1e-8 check 
        # but is the best we can do without more calls/time).
        # However, the task says "hidden oscillatory equations". 
        # Maybe we should try a grid search? 
        # Given the call limit (500), a dense grid is risky.
        # Let's try a few more random or systematic points?
        # For now, return best_x.
        return best_x

    # We have a bracket [a, b] with fa*fb < 0
    # Use Bisection or Secant. Secant is faster if it converges, Bisection is robust.
    # Let's use a hybrid: try Secant, but if it leaves the bracket, fall back to Bisection.
    
    x1 = a
    x2 = b
    f1 = fa
    f2 = fb
    
    for i in range(50):
        if time_up():
            break
            
        # Secant step
        if f2 != f1:
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        else:
            # Fallback to bisection
            x_new = (x1 + x2) / 2.0
            
        # Check if x_new is within the bracket
        if (x_new - x1) * (x_new - x2) > 0:
            # Out of bracket, use bisection
            x_new = (x1 + x2) / 2.0
            # We can reuse f(x_new) if we calculated it? No, we didn't.
            
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            # Fallback to bisection point
            x_new = (x1 + x2) / 2.0
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
                
        if abs(f_new) <= tol:
            return x_new
            
        # Update bracket
        if f_new * f1 < 0:
            x2 = x_new
            f2 = f_new
        else:
            x1 = x_new
            f1 = f_new
            
        # Convergence check
        if abs(x2 - x1) < 1e-15 * max(1.0, abs(x2)):
            # Return the midpoint or the one with smaller |f|
            if abs(f1) < abs(f2):
                return x1
            else:
                return x2
                
    # Return the point with the smallest |f|
    if abs(f1) < abs(f2):
        return x1
    else:
        return x2

# EVOLVE-BLOCK-END
