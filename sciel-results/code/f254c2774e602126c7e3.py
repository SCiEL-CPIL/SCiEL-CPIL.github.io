"""Root finder for smooth polynomial equations with call-count optimization."""
import time

# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 for a smooth polynomial.
    
    Uses Newton's method with secant fallback, optimized for call count.
    """
    deadline = time.time() + 110.0  # 10s safety margin before 120s limit
    
    x = float(x0)
    best_x = x
    best_abs_fx = float('inf')
    
    # Evaluate initial point
    fx = f(x)
    if not math.isfinite(fx):
        # Try a few nearby points
        for dx in [1e-6, -1e-6, 1e-4, -1e-4, 0.1, -0.1]:
            try:
                x_try = x + dx
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    x = x_try
                    fx = fx_try
                    break
            except Exception:
                continue
        else:
            return x  # Give up, return original x0
    
    best_abs_fx = abs(fx)
    if best_abs_fx <= 1e-8:
        return x
    
    # Newton's method with finite-difference derivative
    # For polynomials, this should converge quickly
    x_prev = x
    x_prev2 = x - 1e-4  # for secant fallback
    
    for i in range(50):
        if time.time() > deadline:
            break
        
        if abs(fx) <= 1e-12:
            best_x = x
            break
        
        # Compute derivative via central difference for better accuracy
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
        except Exception:
            break
        
        if not (math.isfinite(fx_plus) and math.isfinite(fx_minus)):
            break
        
        d = (fx_plus - fx_minus) / (2.0 * h)
        
        if d == 0 or not math.isfinite(d):
            # Fall back to secant method
            if i >= 1:
                dx = x - x_prev
                if dx != 0 and math.isfinite(x_prev) and math.isfinite(f(x_prev)):
                    d = (fx - f(x_prev)) / dx
                else:
                    break
            else:
                break
        
        x_new = x - fx / d
        
        # Safeguard: if step is too large, dampen it
        if not math.isfinite(x_new):
            break
        
        step = abs(x_new - x)
        if step > 10.0 * max(1.0, abs(x)):
            # Dampen the step
            x_new = x + 0.5 * (x_new - x) * (1.0 if x_new > x else -1.0)
        
        x_prev2 = x_prev
        x_prev = x
        x = x_new
        
        try:
            fx = f(x)
        except Exception:
            break
        
        if not math.isfinite(fx):
            # Revert and try secant
            x = x_prev
            fx = f(x)
            break
        
        if abs(fx) < best_abs_fx:
            best_abs_fx = abs(fx)
            best_x = x
        
        # Check for convergence
        if abs(fx) <= 1e-8:
            best_x = x
            break
    
    # If Newton didn't converge to 1e-8, try secant method from best point
    if best_abs_fx > 1e-8 and time.time() < deadline:
        x_a = best_x
        fx_a = f(x_a)
        # Get a second point for secant
        h = 1e-3 * max(1.0, abs(x_a))
        x_b = x_a + h
        fx_b = f(x_b)
        
        for i in range(40):
            if time.time() > deadline:
                break
            
            if abs(fx_a) <= 1e-8:
                best_x = x_a
                break
            if abs(fx_b) <= 1e-8:
                best_x = x_b
                break
            
            if fx_a == fx_b:
                break
            
            x_new = x_a - fx_a * (x_b - x_a) / (fx_b - fx_a)
            
            if not math.isfinite(x_new):
                break
            
            # Dampen if step too large
            if abs(x_new - x_a) > 5.0 * max(1.0, abs(x_a)):
                x_new = x_a + 0.5 * (x_new - x_a) * (1.0 if x_new > x_a else -1.0)
            
            x_a, fx_a = x_b, fx_b
            x_b, fx_b = x_new, f(x_new)
            
            if abs(fx_b) < best_abs_fx:
                best_abs_fx = abs(fx_b)
                best_x = x_b
            if abs(fx_a) < best_abs_fx:
                best_abs_fx = abs(fx_a)
                best_x = x_a
    
    return best_x
# EVOLVE-BLOCK-END
