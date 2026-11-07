"""Robust scalar root finder using Illinois/IQI hybrid with bracketing."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a robust bracketing strategy followed by Illinois/Inverse-Quadratic-Interpolation.
    """
    DEADLINE = 148.0  # Safety margin before 150s kill
    START_TIME = time.time()
    
    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE
    
    def is_finite(x):
        return math.isfinite(x)
    
    def safe_f(x):
        """Evaluate f(x) safely, returning None on failure."""
        try:
            fx = f(x)
            if is_finite(fx):
                return fx
        except:
            pass
        return None
    
    def is_good(x, fx):
        """Check if x is a good root."""
        return fx is not None and abs(fx) <= 1e-8

    # Track best solution found so far
    best_x = None
    best_fx = None
    best_abs_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fx, best_abs_fx
        if fx is not None and is_finite(x) and abs(fx) < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs(fx)
            if abs(fx) <= 1e-8:
                return True
        return False

    # Initial evaluation
    fx0 = safe_f(x0)
    if fx0 is not None:
        update_best(x0, fx0)
        if abs(fx0) <= 1e-8:
            return x0

    # Strategy 1: Find a bracket [a, b] where f(a)*f(b) < 0
    a = x0
    fa = fx0
    
    # Try expanding to the right and left
    # Start with a small step
    step = 1e-3 * max(1.0, abs(x0))
    b = x0 + step
    fb = safe_f(b)
    
    bracket_found = False
    if fb is not None:
        update_best(b, fb)
        if fa * fb < 0:
            bracket_found = True
        else:
            # Expand b
            for _ in range(50):
                if not check_deadline():
                    break
                step *= 2.0
                b = x0 + step
                fb = safe_f(b)
                if fb is None:
                    break
                if update_best(b, fb):
                    return b
                if fa * fb < 0:
                    bracket_found = True
                    break
                # If f is getting very large, stop expanding this direction
                if abs(fb) > 1e100:
                    break
            
            if not bracket_found:
                # Try expanding to the left
                step = 1e-3 * max(1.0, abs(x0))
                a = x0 - step
                fa = safe_f(a)
                if fa is not None:
                    update_best(a, fa)
                    if fa * fx0 < 0:
                        # We have a bracket [a, x0]
                        b = x0
                        fb = fx0
                        bracket_found = True
                    else:
                        # Expand a
                        for _ in range(50):
                            if not check_deadline():
                                break
                            step *= 2.0
                            a = x0 - step
                            fa = safe_f(a)
                            if fa is None:
                                break
                            if update_best(a, fa):
                                return a
                            if fa * fx0 < 0:
                                b = x0
                                fb = fx0
                                bracket_found = True
                                break
                            if abs(fa) > 1e100:
                                break
    
    if not bracket_found:
        # If we couldn't find a bracket, try a few secant steps from x0
        # This is a fallback for cases where the root is not bracketed easily
        # but secant might converge.
        x1 = x0
        fx1 = fx0
        x2 = x0 + 1e-3 * max(1.0, abs(x0))
        fx2 = safe_f(x2)
        
        if fx2 is None:
            if best_x is not None:
                return best_x
            return x0
        
        update_best(x2, fx2)
        
        for _ in range(100):
            if not check_deadline():
                break
            if fx1 == fx2:
                break
            x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            if not is_finite(x_new):
                break
            x1 = x2
            fx1 = fx2
            x2 = x_new
            fx2 = safe_f(x2)
            if fx2 is None:
                break
            if update_best(x2, fx2):
                return x2
        
        if best_x is not None:
            return best_x
        return x0

    # Strategy 2: Illinois Algorithm (Weighted Bisection)
    # This is robust and converges faster than bisection.
    # It maintains the invariant f(a)*f(b) < 0.
    
    # Ensure a is the "smaller" magnitude side initially? No, Illinois just swaps.
    # Standard Illinois:
    # 1. Evaluate midpoint m.
    # 2. If f(a)*f(m) < 0, b = m, else a = m.
    # 3. If |f(a)| < |f(b)|, shrink the side with larger |f| by halving its weight.
    
    # Initialize weights
    wa = 1.0
    wb = 1.0
    
    for iteration in range(1000):
        if not check_deadline():
            break
        
        # Weighted midpoint
        if wa == wb:
            m = (a + b) / 2.0
        else:
            m = (wa * a + wb * b) / (wa + wb)
        
        fm = safe_f(m)
        if fm is None:
            # If evaluation fails, fall back to standard bisection step
            m = (a + b) / 2.0
            fm = safe_f(m)
            if fm is None:
                break
        
        if update_best(m, fm):
            return m
        
        if fa * fm < 0:
            b = m
            fb = fm
            wb = 1.0
        else:
            a = m
            fa = fm
            wa = 1.0
        
        # Illinois weighting: shrink the weight of the side with larger |f|
        if abs(fa) < abs(fb):
            wb *= 0.5
        else:
            wa *= 0.5
        
        # Check for convergence of the bracket
        if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
            break

    # Strategy 3: Polish with Secant Method
    # Use the last two points from the Illinois run (or best available) to do secant steps.
    # The Illinois run ends with a very small bracket. The best_x is likely very close.
    # Let's use the last two distinct points evaluated to start secant.
    
    # We need two points. a and b are the current bracket ends.
    # f(a) and f(b) are known.
    x1 = a
    fx1 = fa
    x2 = b
    fx2 = fb
    
    if x1 == x2 or fx1 == fx2:
        # Degenerate case, return best
        if best_x is not None:
            return best_x
        return m # last midpoint
    
    for _ in range(50):
        if not check_deadline():
            break
        
        # Secant step
        if fx1 == fx2:
            break
        x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
        
        if not is_finite(x_new):
            break
            
        # Check if x_new is close to x1 or x2 (stalling)
        if abs(x_new - x1) < 1e-15 * max(1.0, abs(x1)) and abs(x_new - x2) < 1e-15 * max(1.0, abs(x2)):
            break

        x1 = x2
        fx1 = fx2
        x2 = x_new
        fx2 = safe_f(x2)
        
        if fx2 is None:
            break
        
        if update_best(x2, fx2):
            return x2
            
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
