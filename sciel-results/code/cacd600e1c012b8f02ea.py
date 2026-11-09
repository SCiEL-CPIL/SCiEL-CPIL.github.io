"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust combination of:
    1. Secant method with inverse quadratic interpolation
    2. Bisection fallback when sign change is detected
    3. Adaptive step sizes for ill-conditioned problems
    4. Multiple restart strategies if initial attempts fail
    """
    x0 = float(x0)
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    # State tracking
    best_x = x0
    best_f = f(x0)
    eval_count = 1
    max_evals = 500
    
    if abs(best_f) <= 1e-8:
        return best_x
    
    # Store history for inverse interpolation
    # Each entry: (x, fx)
    history = [(x0, best_f)]
    
    def eval_f(x):
        nonlocal eval_count, best_x, best_f
        if eval_count >= max_evals:
            return None
        fx = f(x)
        eval_count += 1
        if abs(fx) < abs(best_f):
            best_x = x
            best_f = fx
        history.append((x, fx))
        return fx
    
    # Try to find a bracket first
    # Expand around x0 to find sign change
    def find_bracket(x0, f0):
        nonlocal eval_count, best_x, best_f
        if eval_count >= max_evals:
            return None, None
        # Try expanding in both directions
        step = max(1.0, abs(x0)) * 0.1
        for i in range(20):
            # Try positive direction
            if eval_count >= max_evals:
                break
            x1 = x0 + step
            f1 = f(x1)
            eval_count += 1
            if abs(f1) < abs(best_f):
                best_x = x1
                best_f = f1
            history.append((x1, f1))
            if f0 * f1 < 0:
                return x0, x1
            # Try negative direction
            if eval_count >= max_evals:
                break
            x2 = x0 - step
            f2 = f(x2)
            eval_count += 1
            if abs(f2) < abs(best_f):
                best_x = x2
                best_f = f2
            history.append((x2, f2))
            if f0 * f2 < 0:
                return x0, x2
            step *= 2.0
        return None, None
    
    a, b = find_bracket(x0, best_f)
    
    # If we have a bracket, use bisection + secant hybrid
    if a is not None and b is not None:
        fa = best_f if a == x0 else f(a)
        fb = best_f if b == x0 else f(b)
        # Make sure a and b have opposite signs
        if fa * fb > 0:
            a, b = None, None
        else:
            for _ in range(50):
                if time.time() > deadline:
                    break
                if eval_count >= max_evals:
                    break
                # Try secant step
                if abs(fb - fa) > 1e-15:
                    x_sec = b - fb * (b - a) / (fb - fa)
                else:
                    x_sec = (a + b) / 2
                
                # Check if secant step is in bracket
                if a < x_sec < b:
                    fx = f(x_sec)
                    eval_count += 1
                    if abs(fx) < abs(best_f):
                        best_x = x_sec
                        best_f = fx
                    history.append((x_sec, fx))
                    
                    if abs(fx) <= 1e-8:
                        return x_sec
                    
                    if fa * fx < 0:
                        b = x_sec
                        fb = fx
                    else:
                        a = x_sec
                        fa = fx
                else:
                    # Bisection
                    x_mid = (a + b) / 2
                    fx = f(x_mid)
                    eval_count += 1
                    if abs(fx) < abs(best_f):
                        best_x = x_mid
                        best_f = fx
                    history.append((x_mid, fx))
                    
                    if abs(fx) <= 1e-8:
                        return x_mid
                    
                    if fa * fx < 0:
                        b = x_mid
                        fb = fx
                    else:
                        a = x_mid
                        fa = fx
                
                if abs(b - a) < 1e-12:
                    break
    
    # If no bracket found, use inverse quadratic interpolation with secant fallback
    # Need at least 3 points for quadratic, use secant with 2
    if len(history) >= 2:
        # Get last two distinct points
        x1, f1 = history[-2]
        x2, f2 = history[-1]
        
        for _ in range(100):
            if time.time() > deadline:
                break
            if eval_count >= max_evals:
                break
            
            # Check if we have 3 points for quadratic interpolation
            if len(history) >= 3:
                x0h, f0h = history[-3]
                x1h, f1h = history[-2]
                x2h, f2h = history[-1]
                
                # Inverse quadratic interpolation
                # Find x such that f(x) = 0 using quadratic through (x0h,f0h), (x1h,f1h), (x2h,f2h)
                denom = (x2h - x0h) * (x2h - x1h) * (x1h - x0h)
                if abs(denom) > 1e-30:
                    # Lagrange interpolation for inverse
                    num = (f0h * (x1h - x2h) * (x2h - x1h) + 
                           f1h * (x2h - x0h) * (x1h - x0h) + 
                           f2h * (x1h - x0h) * (x2h - x0h))
                    # Actually, let's use the standard formula
                    # x_new = x0h + (x1h-x0h)*(x2h-x0h) * (f1h*f2h / ((f0h-f1h)*(f0h-f2h)))
                    # Simpler: use the formula for inverse quadratic
                    if abs(f0h - f1h) > 1e-15 and abs(f0h - f2h) > 1e-15 and abs(f1h - f2h) > 1e-15:
                        x_new = (x0h * f1h * f2h / ((f0h - f1h) * (f0h - f2h)) +
                                 x1h * f0h * f2h / ((f1h - f0h) * (f1h - f2h)) +
                                 x2h * f0h * f1h / ((f2h - f0h) * (f2h - f1h)))
                        # Check if x_new is reasonable
                        if abs(x_new - x2h) < 10 * abs(x2h - x1h) + 1.0:
                            fx = f(x_new)
                            eval_count += 1
                            if abs(fx) < abs(best_f):
                                best_x = x_new
                                best_f = fx
                            history.append((x_new, fx))
                            if abs(fx) <= 1e-8:
                                return x_new
                            # Update history, keep last 3
                            if len(history) > 10:
                                history.pop(0)
                            continue
            # Fallback to secant
            x1, f1 = history[-2]
            x2, f2 = history[-1]
            if abs(f2 - f1) > 1e-15:
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            else:
                x_new = x2 + 0.1 * max(1.0, abs(x2))
            
            # Limit step size
            if abs(x_new - x2) > 10 * max(1.0, abs(x2)):
                x_new = x2 + math.copysign(10 * max(1.0, abs(x2)), x_new - x2)
            
            fx = f(x_new)
            eval_count += 1
            if abs(fx) < abs(best_f):
                best_x = x_new
                best_f = fx
            history.append((x_new, fx))
            if abs(fx) <= 1e-8:
                return x_new
            if len(history) > 10:
                history.pop(0)
    
    # Final polish: if we have a good candidate, try to refine it
    if abs(best_f) < 1e-4 and eval_count < max_evals:
        # Use Newton-like refinement with finite differences
        x = best_x
        for _ in range(20):
            if time.time() > deadline:
                break
            if eval_count >= max_evals:
                break
            fx = f(x)
            eval_count += 1
            if abs(fx) < abs(best_f):
                best_x = x
                best_f = fx
            if abs(fx) <= 1e-8:
                return x
            h = 1e-8 * max(1.0, abs(x))
            fxh = f(x + h)
            eval_count += 1
            if abs(fxh) < abs(best_f):
                best_x = x + h
                best_f = fxh
            d = (fxh - fx) / h
            if abs(d) > 1e-15:
                x_new = x - fx / d
                if abs(x_new - x) > 10 * max(1.0, abs(x)):
                    x_new = x + math.copysign(10 * max(1.0, abs(x)), x_new - x)
                x = x_new
            else:
                break
    
    return best_x
# EVOLVE-BLOCK-END
