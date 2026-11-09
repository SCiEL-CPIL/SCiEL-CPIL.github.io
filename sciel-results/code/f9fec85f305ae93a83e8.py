import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    max_calls = 450
    call_count = 0

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    if not math.isfinite(x):
        return x

    fx = f_call(x)

    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    # Step 1: Find a bracket by expanding in both directions
    a = x
    fa = fx
    b = None
    fb = None

    # Try expanding right
    h = 0.5
    for _ in range(40):
        if not time_ok():
            break
        b = a + h
        fb = f_call(b)
        if not math.isfinite(fb):
            h *= 2.0
            continue
        if fa * fb < 0:
            break
        a = b
        fa = fb
        h *= 2.0

    if b is None or fa * fb >= 0:
        # Try expanding left from original x0
        a = x
        fa = fx
        b = None
        fb = None
        h = 0.5
        for _ in range(40):
            if not time_ok():
                break
            b = a - h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if fa * fb < 0:
                break
            a = b
            fa = fb
            h *= 2.0

    if b is not None and fa * fb < 0:
        # We have a bracket [a, b] with f(a)*f(b) < 0
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Use Brent's method for fast convergence
        # Brent's method combines bisection, secant, and inverse quadratic interpolation
        # It guarantees linear convergence and achieves superlinear convergence near the root
        
        # Initialize
        c = a
        fc = fa
        d = b - a
        e = d
        mflag = True
        
        # Track the best point found
        best_x = x
        best_f = fx
        
        for _ in range(200):
            if not time_ok():
                break
            
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            if abs(fc) <= 1e-8:
                return c
            
            # Update best
            if abs(fb) < abs(best_f):
                best_x = b
                best_f = fb
            if abs(fa) < abs(best_f):
                best_x = a
                best_f = fa
            
            # Check convergence
            if abs(b - a) < 1e-13 * max(1.0, abs(a), abs(b)):
                # Return the point with smallest |f|
                candidates = [(a, fa), (b, fb), (c, fc)]
                best = min(candidates, key=lambda t: abs(t[1]))
                return best[0]
            
            # Inverse quadratic interpolation
            if mflag:
                # Use inverse quadratic interpolation if possible
                if fa != fb and fb != fc:
                    # Compute inverse quadratic interpolation
                    x1 = b
                    x2 = c
                    x3 = a
                    f1 = fb
                    f2 = fc
                    f3 = fa
                    
                    # p = f2*f3/(f1-f3) * (x1-x3) + f1*f2/(f1-f2) * (x1-x2) ... wait, let me use the standard formula
                    # Actually, let's use the standard Brent's method formulation
                    # s = (f2*f3/(f1-f3))*(x1-x3) + (f1*f3/(f1-f2))*(x1-x2) ... no
                    # Let me use the correct formula:
                    # p = f2*f3/(f1-f3) * (x1-x3) + f1*f2/(f1-f2) * (x1-x2) is not right
                    # The standard is:
                    # s = x1 + (x1-x2)*f1*f3/((f1-f2)*(f1-f3)) + (x1-x3)*f1*f2/((f1-f3)*(f2-f3))
                    # Let me just use the secant method as fallback for simplicity and reliability
                    
                    # Actually, let me implement a simpler but robust version:
                    # Use secant when it stays in bracket, otherwise bisect
                    pass
                
                # Secant step
                if fb != fa:
                    x_secant = b - fb * (b - a) / (fb - fa)
                else:
                    x_secant = 0.5 * (a + b)
                
                # Use secant if it's inside the bracket, otherwise bisect
                if a < x_secant < b:
                    s = x_secant
                else:
                    s = 0.5 * (a + b)
                    mflag = False
            else:
                # Just bisect
                s = 0.5 * (a + b)
            
            fs = f_call(s)
            
            if not math.isfinite(fs):
                # Fallback to bisection
                s = 0.5 * (a + b)
                fs = f_call(s)
            
            if abs(fs) <= 1e-8:
                return s
            
            # Update best
            if abs(fs) < abs(best_f):
                best_x = s
                best_f = fs
            
            # Update bracket
            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs
            
            # Update c
            if abs(fb) < abs(fc):
                c = a
                fc = fa
            else:
                c = s
                fc = fs
            
            d = b - a
            e = d
            
            mflag = True

        # Return the best point found
        return best_x

    else:
        # No bracket found. Try secant method with finite differences for derivative-free approach
        # Use two points and iterate
        x1 = x
        f1 = fx
        h = 1e-6 * max(1.0, abs(x))
        x2 = x1 + h
        f2 = f_call(x2)
        
        if not math.isfinite(f2):
            h = 1e-6 * max(1.0, abs(x))
            x2 = x1 - h
            f2 = f_call(x2)
        
        if not math.isfinite(f2):
            return x1
        
        best_x = x1 if abs(f1) < abs(f2) else x2
        best_f = min(abs(f1), abs(f2))
        
        for _ in range(100):
            if not time_ok():
                break
            
            if abs(f2) <= 1e-8:
                return x2
            if abs(f1) <= 1e-8:
                return x1
            
            if abs(f2) < best_f:
                best_x = x2
                best_f = abs(f2)
            if abs(f1) < best_f:
                best_x = x1
                best_f = abs(f1)
            
            # Secant step
            if f2 != f1:
                x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            else:
                break
            
            if not math.isfinite(x3):
                break
            
            # Damping: limit step size
            step = x3 - x2
            if abs(step) > 10.0 * max(1.0, abs(x2)):
                x3 = x2 + 10.0 * max(1.0, abs(x2)) * math.copysign(1.0, step)
            
            f3 = f_call(x3)
            
            if not math.isfinite(f3):
                break
            
            if abs(f3) < best_f:
                best_x = x3
                best_f = abs(f3)
            
            # Move forward
            x1, f1 = x2, f2
            x2, f2 = x3, f3
        
        return best_x
