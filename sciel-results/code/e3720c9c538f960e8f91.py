import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    max_calls = 480
    call_count = 0
    
    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    # Helper to call f and track count
    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    fx = f_call(x)
    
    if not math.isfinite(fx):
        return x # Or handle error, but task says non-finite x is failure. 
                 # Returning x if f(x) is non-finite might be bad. 
                 # But if f(x0) is inf, we are stuck. 
                 # Let's assume f(x0) is finite.
    
    if abs(fx) <= 1e-8:
        return x

    # Try to find a bracket
    a = x
    fa = fx
    # Try expanding to the right first
    h = 1.0
    b = None
    fb = None
    
    for _ in range(20):
        if not time_ok():
            break
        b = a + h
        fb = f_call(b)
        if not math.isfinite(fb):
            h *= 2
            continue
        if fa * fb < 0:
            # Bracket found [a, b]
            break
        # If no sign change, move a to b and expand
        a = b
        fa = fb
        h *= 2
        
    if b is not None and fa * fb < 0:
        # We have a bracket [a, b]
        # Use Illinois/Secant-Bisection hybrid
        # Keep track of last two points for Secant
        # Let's use a simple robust loop
        # We know f(a) and f(b)
        
        # Initialize for Secant: we need two points. 
        # We have a and b. 
        # Let's just iterate.
        # To use Secant, we need a previous estimate. 
        # We can use the midpoint as the first estimate, then use Secant with (a, mid) or (b, mid).
        
        # Let's use a standard robust approach:
        # 1. Try Secant.
        # 2. If it fails (outside bracket), use Bisection.
        
        # We have a, fa, b, fb.
        # Let's pick a point.
        # If |fa| < |fb|, we are closer to root at a.
        
        # Let's use the Illinois algorithm properly.
        # It maintains a, b, and the last point p.
        # Actually, simpler:
        # Just alternate Secant and Bisection? No.
        
        # Let's implement a simple "Secant if inside, Bisection if outside"
        # We need a previous point 'p' to form a secant line with the other end.
        # Let's start with p = (a+b)/2.
        p = 0.5 * (a + b)
        fp = f_call(p)
        
        if abs(fp) <= 1e-8:
            return p
            
        # Now we have a, b, p.
        # We can use Secant between (p, a) or (p, b).
        # Let's maintain the two most recent points for Secant.
        x_prev = a
        fx_prev = fa
        x_curr = p
        fx_curr = fp
        
        # Update bracket
        if fx_prev * fx_curr < 0:
            b = x_curr
            fb = fx_curr
        else:
            a = x_curr
            fa = fx_curr
            
        for _ in range(100):
            if not time_ok():
                break
                
            # Check convergence
            if abs(fx_curr) <= 1e-8:
                return x_curr
            if abs(fx_prev) <= 1e-8:
                return x_prev
                
            # Check if interval is small enough
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                return 0.5 * (a + b)

            # Try Secant
            # Secant formula: x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            if fx_curr != fx_prev:
                x_secant = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            else:
                x_secant = 0.5 * (a + b)
                
            # Check if Secant step is inside bracket [a, b]
            if a < x_secant < b:
                x_next = x_secant
            else:
                # Use Bisection
                x_next = 0.5 * (a + b)
                
            f_next = f_call(x_next)
            
            if not math.isfinite(f_next):
                # If we get a non-finite value, shrink to bisection
                x_next = 0.5 * (a + b)
                f_next = f_call(x_next)
                
            if abs(f_next) <= 1e-8:
                return x_next
                
            # Update points for next Secant iteration
            # We want to keep the two most recent points that are "active"
            # Usually, we keep the new point and the point that is still in the bracket
            # or just the two most recent evaluations.
            # Standard Secant keeps last two.
            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_next
            fx_curr = f_next
            
            # Update Bracket
            if fa * f_next < 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next
                
        return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev

    else:
        # No bracket found. Fall back to Newton's Method with Finite Difference
        # This is risky but better than nothing.
        x = float(x0)
        fx = f_call(x)
        
        for _ in range(60):
            if not time_ok():
                break
            if abs(fx) <= 1e-8:
                return x
                
            h = 1e-6 * max(1.0, abs(x))
            fx_plus = f_call(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or not math.isfinite(d):
                break
                
            x_new = x - fx / d
            
            if not math.isfinite(x_new):
                break
                
            # Damping if step is too large
            if abs(x_new - x) > 10.0:
                x_new = x + 10.0 * math.copysign(1.0, x_new - x)
                
            x = x_new
            fx = f_call(x)
            
        return x
