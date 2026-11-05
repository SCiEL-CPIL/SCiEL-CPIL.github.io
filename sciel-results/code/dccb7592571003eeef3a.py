"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Strategy:
    1. Evaluate f(x0). If converged, return.
    2. Attempt to find a bracket [a, b] where f(a)f(b) < 0.
    3. If a bracket is found, use Brent's Method (Bisection + Inverse Quadratic Interpolation)
       for fast and robust convergence.
    4. If no bracket is found, use Damped Newton/Secant methods to minimize |f(x)|.
    5. Return the best x found (min |f(x)|) if no exact root is found.
    """
    TOL = 1e-8
    MAX_EVALS = 480  # Leave some margin
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    def call_f(x):
        """Evaluate f with counting, deadline check, and best-tracking."""
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, FloatingPointError):
            return None
        if not math.isfinite(val):
            return None
        abs_val = abs(val)
        if abs_val < best_abs:
            best_abs = abs_val
            best_x = x
            best_f = val
        return val
    
    def is_good(fx):
        return fx is not None and abs(fx) <= TOL

    # Initial evaluation
    x = float(x0)
    fx = call_f(x)
    if fx is None:
        return best_x
    if is_good(fx):
        return x

    # --- Phase 1: Try to find a bracket ---
    # We want a and b such that f(a)*f(b) < 0
    a = x
    fa = fx
    
    # Initial step size
    # Use a small step to catch local behavior, then expand
    scale = max(1.0, abs(x))
    delta = 1e-4 * scale
    
    bracket_found = False
    b = None
    fb = None
    
    # Try expanding in both directions
    for i in range(50):
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS:
            break
            
        # Try positive direction
        b_try = a + delta * (i + 1)
        fb_try = call_f(b_try)
        if fb_try is not None:
            if is_good(fb_try):
                return b_try
            if fa * fb_try < 0:
                b = b_try
                fb = fb_try
                bracket_found = True
                break
        
        # Try negative direction
        b_try2 = a - delta * (i + 1)
        fb_try2 = call_f(b_try2)
        if fb_try2 is not None:
            if is_good(fb_try2):
                return b_try2
            if fa * fb_try2 < 0:
                b = b_try2
                fb = fb_try2
                bracket_found = True
                break
        
        # Expand delta exponentially
        delta *= 2.0
        
        # If we've expanded a lot, maybe try a different center?
        # For now, stick to x0 centered expansion.

    if bracket_found:
        # --- Phase 2: Brent's Method ---
        # a, b are the bracket. fa, fb are the function values.
        # We will maintain c as the previous b for IQI.
        c = a
        fc = fa
        d = 0.0  # Previous step
        e = 0.0  # Step size of the previous step
        
        for _ in range(100):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS:
                break
            
            if is_good(fa):
                return a
            if is_good(fb):
                return b
            
            # Ensure a and b are ordered for clarity, but Brent works with signs
            # Let's keep a as the "old" b and b as the "new" b?
            # Standard Brent:
            # 1. If fa and fb have same sign, something is wrong (should not happen if maintained)
            # 2. Swap if |fb| < |fa|
            if abs(fb) < abs(fa):
                a, b = b, a
                fa, fb = fb, fa
            
            # c is the previous a (before swap)
            # Actually, standard algorithm:
            # c = a (old a)
            # fc = fa (old fa)
            # Let's restart the logic slightly to be safe.
            
            # Inverse Quadratic Interpolation (IQI) or Secant
            # Points: (a, fa), (b, fb), (c, fc)
            # c is the point before a (in terms of history)
            # We need to store history properly.
            
            # Let's use a simpler robust hybrid: Safeguarded Secant/Bisection
            # This is effectively what Brent does but easier to implement correctly.
            
            # Secant step
            if fa != fb:
                x_new = b - fb * (b - a) / (fb - fa)
            else:
                x_new = (a + b) / 2.0
            
            # Check if x_new is within [a, b]
            if (a < x_new < b) or (b < x_new < a):
                # It's inside the bracket
                pass
            else:
                # Fall back to bisection
                x_new = (a + b) / 2.0
            
            # Additional safeguard: if x_new is too close to a or b, use bisection
            # to ensure progress
            if abs(x_new - a) < 1e-12 * max(1.0, abs(a)) or abs(x_new - b) < 1e-12 * max(1.0, abs(b)):
                x_new = (a + b) / 2.0
                
            f_new = call_f(x_new)
            if f_new is None:
                break
            
            if is_good(f_new):
                return x_new
                
            # Update bracket
            if fa * f_new <= 0:
                # Root is in [a, x_new]
                b = x_new
                fb = f_new
            else:
                # Root is in [x_new, b]
                a = x_new
                fa = f_new
                
            # Convergence check on interval
            if abs(b - a) < 1e-12 * max(1.0, abs(x_new)):
                # Return the point with smaller |f|
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b

        # Return best from bracket
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    # --- Phase 3: No Bracket Found ---
    # Use Damped Newton / Secant to minimize |f(x)|
    # We have x, fx.
    # Try Secant method
    x1 = x
    fx1 = fx
    x2 = x + 1e-4 * max(1.0, abs(x))
    fx2 = call_f(x2)
    
    if fx2 is None:
        # Try negative direction
        x2 = x - 1e-4 * max(1.0, abs(x))
        fx2 = call_f(x2)
        
    if fx2 is None:
        return best_x

    for _ in range(50):
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS:
            break
            
        if is_good(fx1):
            return x1
        if is_good(fx2):
            return x2
            
        if fx1 == fx2:
            break
            
        # Secant step
        x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
        
        # Damping: if step is huge, damp it
        step = x_new - x2
        max_step = 10.0 * max(1.0, abs(x2))
        if abs(step) > max_step:
            x_new = x2 + math.copysign(max_step, step)
            
        fx_new = call_f(x_new)
        if fx_new is None:
            break
            
        if is_good(fx_new):
            return x_new
            
        # Update points: keep the two points with smallest |f|? 
        # Or just shift window?
        # Standard secant shifts window.
        # But for ill-conditioned, maybe keep the best?
        # Let's stick to standard secant shift, but track global best.
        x1, fx1 = x2, fx2
        x2, fx2 = x_new, fx_new

    return best_x

# EVOLVE-BLOCK-END
