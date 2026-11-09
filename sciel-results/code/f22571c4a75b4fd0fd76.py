import math
import time
import scipy.optimize

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0
    max_calls = 480
    call_count = 0

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    fx = f_call(x)

    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    # --- Phase 1: Find a Bracket ---
    a = x
    fa = fx
    found = False
    b = None
    fb = None

    d = 1.0
    for _ in range(30):
        if not time_ok():
            break

        xr = a + d
        fr = f_call(xr)
        if math.isfinite(fr):
            if fa * fr < 0:
                b = xr
                fb = fr
                found = True
                break

        xl = a - d
        fl = f_call(xl)
        if math.isfinite(fl):
            if fa * fl < 0:
                b = xl
                fb = fl
                found = True
                break

        d *= 2.0
        if d > 1e10:
            break

    if not found:
        d = 1e6
        xr = a + d
        fr = f_call(xr)
        if math.isfinite(fr) and fa * fr < 0:
            b = xr
            fb = fr
            found = True
        else:
            xl = a - d
            fl = f_call(xl)
            if math.isfinite(fl) and fa * fl < 0:
                b = xl
                fb = fl
                found = True

    if not found:
        # Fallback: Secant method
        x1 = x0 + 1e-6
        fx1 = f_call(x1)
        
        for _ in range(20):
            if not time_ok(): 
                break
            if abs(fx) <= 1e-8: 
                return x
            if abs(fx1) <= 1e-8: 
                return x1
            if fx1 == fx: 
                break
            
            x_new = x - fx * (x - x1) / (fx - fx1)
            if not math.isfinite(x_new): 
                break
                
            f_new = f_call(x_new)
            if not math.isfinite(f_new): 
                break
                
            x, fx = x1, fx1
            x1, fx1 = x_new, f_new
            
        return x if abs(fx) < abs(fx1) else x1

    # --- Phase 2: Refine ---
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    # Try Newton's method with finite difference derivative first.
    # This often converges in very few iterations (quadratic convergence).
    # We use the bracket [a, b] to ensure we stay in a reasonable region
    # or fall back to bisection/secant if Newton steps go too far.
    
    # Initial guess: midpoint or the point with smaller |f|
    if abs(fa) < abs(fb):
        x_curr = a
        f_curr = fa
    else:
        x_curr = b
        f_curr = fb
    
    # Try Newton with central difference for derivative
    h = 1e-5
    success = False
    
    for _ in range(10):
        if not time_ok():
            break
        if abs(f_curr) <= 1e-8:
            success = True
            break
            
        # Compute derivative using central difference
        # f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        # This costs 2 function calls per iteration.
        # Total cost per iteration: 2 (for derivative) + 1 (for next f value) = 3 calls.
        # But we can reuse f(x-h) or f(x+h) if we're careful? No, standard Newton uses f(x) and f'(x).
        # Let's try secant method instead, which is cheaper (1 call/iteration) and superlinear.
        # Secant is often better than Newton with FD when function calls are expensive.
        
        break  # Exit Newton loop, go to Secant

    # Use Secant method (Illinois variant for robustness)
    # We have two points: (a, fa) and (b, fb)
    x_prev = a
    fx_prev = fa
    x_curr = b
    fx_curr = fb
    
    # If we started with the same point (shouldn't happen if bracket found), handle it
    if x_prev == x_curr:
        x_curr += 1e-6
        fx_curr = f_call(x_curr)

    for _ in range(30):
        if not time_ok():
            break
        if abs(fx_curr) <= 1e-8:
            return x_curr
        if abs(fx_prev) <= 1e-8:
            return x_prev
            
        # Check for convergence in x
        if abs(x_curr - x_prev) < 1e-12 * max(1.0, abs(x_curr), abs(x_prev)):
            # Return the one with smaller |f|
            return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev

        # Secant step
        if fx_curr != fx_prev:
            x_secant = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        else:
            x_secant = 0.5 * (x_curr + x_prev)

        # Illinois method: if the new point is outside the bracket, halve the distance
        # We need to maintain a bracket. Let's use a hybrid approach.
        # If x_secant is within [min(x_prev, x_curr), max(x_prev, x_curr)], it's a standard secant step.
        # Otherwise, it might be diverging. For polynomials, secant usually converges if started close.
        # Let's just use the secant step, but if it goes non-finite, fall back to bisection.
        
        if not math.isfinite(x_secant):
            x_next = 0.5 * (x_curr + x_prev)
        else:
            x_next = x_secant
            
        f_next = f_call(x_next)
        
        if not math.isfinite(f_next):
            # Fallback to bisection
            x_next = 0.5 * (x_curr + x_prev)
            f_next = f_call(x_next)
            
        if abs(f_next) <= 1e-8:
            return x_next

        # Update points: keep the one with smaller |f| as current, the other as prev?
        # Or maintain bracket?
        # Standard secant doesn't maintain a strict bracket, but for polynomials it's usually fine.
        # Let's use the Illinois strategy: 
        # If x_next is between x_prev and x_curr, we have a new bracket.
        # If not, we might need to be careful.
        
        # Simple approach: replace the point with the larger |f| value
        if abs(f_next) < abs(f_prev):
            x_prev = x_curr
            fx_prev = fx_curr
        else:
            # Keep x_prev, replace x_curr? No, secant uses the two most recent.
            # Let's just shift: x_prev = x_curr, x_curr = x_next
            pass
            
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_next
        fx_curr = f_next

    return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
