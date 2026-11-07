"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    x = float(x0)
    fx = f(x)
    best_x = x
    best_abs_fx = abs(fx)
    
    if best_abs_fx <= 1e-8:
        return x
    
    # Newton's method with central difference derivative and damping
    for iteration in range(80):
        if time.time() > deadline:
            break
        if best_abs_fx <= 1e-12:
            return best_x
        
        h = 1e-7 * max(1.0, abs(x))
        # Central difference for better derivative estimate
        d = (f(x + h) - f(x - h)) / (2 * h)
        
        if d == 0 or not math.isfinite(d):
            break
        
        newton_step = fx / d
        # Damp the step if it's too large
        if abs(newton_step) > 10.0 * max(1.0, abs(x)):
            newton_step = math.copysign(10.0 * max(1.0, abs(x)), newton_step)
        
        x_new = x - newton_step
        
        if not math.isfinite(x_new):
            break
        
        fx_new = f(x_new)
        
        if not math.isfinite(fx_new):
            break
        
        abs_fx_new = abs(fx_new)
        if abs_fx_new < best_abs_fx:
            best_x = x_new
            best_abs_fx = abs_fx_new
        
        if abs_fx_new <= 1e-12:
            return x_new
        
        x = x_new
        fx = fx_new
    
    # Bisection fallback: find a bracket where sign changes
    if time.time() > deadline:
        return best_x
    
    # Try to find a bracket around x0 or best_x
    def find_bracket(func, center, f_center):
        """Find [a, b] where func(a) and func(b) have opposite signs."""
        if not math.isfinite(f_center):
            return None
        if abs(f_center) <= 1e-8:
            return (center, center)
        
        step = max(1e-3, abs(center) * 0.1)
        a = center
        fa = f_center
        b = center + step
        fb = func(b)
        
        # Expand bracket to find sign change
        max_expansions = 50
        for _ in range(max_expansions):
            if time.time() > deadline:
                return None
            if not math.isfinite(fb):
                b = b + step
                fb = func(b)
                continue
            if fa * fb <= 0:
                return (a, b)
            a = b
            fa = fb
            b = b + step
            fb = func(b)
        
        # Try expanding in the other direction
        a = center - step
        fa = func(a)
        b = center
        fb = f_center
        for _ in range(max_expansions):
            if time.time() > deadline:
                return None
            if not math.isfinite(fa):
                a = a - step
                fa = func(a)
                continue
            if fa * fb <= 0:
                return (a, b)
            b = a
            fb = fa
            a = a - step
            fa = func(a)
        
        return None
    
    # Try bracketing from best_x first, then x0
    bracket = None
    if math.isfinite(best_abs_fx) and best_abs_fx < 1e-2:
        bracket = find_bracket(f, best_x, f(best_x))
    
    if bracket is None:
        bracket = find_bracket(f, x, fx)
    
    if bracket is not None:
        a, b = bracket
        fa = f(a) if a != b else f(a)
        fb = f(b) if b != a else f(b)
        
        # Bisection
        for _ in range(100):
            if time.time() > deadline:
                break
            mid = (a + b) / 2.0
            fm = f(mid)
            
            if not math.isfinite(fm):
                break
            
            abs_fm = abs(fm)
            if abs_fm < best_abs_fx:
                best_x = mid
                best_abs_fx = abs_fm
            
            if abs_fm <= 1e-12:
                return mid
            
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            
            if abs(b - a) < 1e-14:
                break
    
    return best_x
# EVOLVE-BLOCK-END
