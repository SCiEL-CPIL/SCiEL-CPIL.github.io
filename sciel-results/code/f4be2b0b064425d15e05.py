"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f near x0 using Inverse Quadratic Interpolation (IQI)
    with Bisection safeguard.
    """
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # 1. Robust Bracketing
    a = x
    fa = fx
    b = None
    fb = None
    bracket_found = False
    
    # Try to find a bracket quickly
    step = 1e-4
    for i in range(50):
        if i % 2 == 0:
            b = a + step
        else:
            b = a - step
            
        fb = f(b)
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0:
            bracket_found = True
            break
        step *= 2.0
        if not math.isfinite(b):
            break

    if not bracket_found or not math.isfinite(b):
        # Fallback: Secant method
        x1 = a
        f1 = fa
        x2 = a + 1e-4
        f2 = f(x2)
        for _ in range(50):
            if abs(f2) <= 1e-8: return x2
            if abs(f1) <= 1e-8: return x1
            if f1 == f2: break
            x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x_new): break
            f_new = f(x_new)
            if abs(f_new) <= 1e-8: return x_new
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
        return x1 if abs(f1) < abs(f2) else x2

    # 2. IQI Main Loop
    # We have bracket [a, b] with fa, fb.
    # Need a third point c. Use midpoint.
    c = (a + b) / 2.0
    fc = f(c)
    if abs(fc) <= 1e-8:
        return c

    # State: a, b are endpoints. c is the last interior point.
    # The 3 points for IQI are (a, fa), (b, fb), (c, fc).
    
    for _ in range(100):
        # Calculate IQI estimate
        x_new = None
        denom1 = (fa - fb) * (fa - fc)
        denom2 = (fb - fa) * (fb - fc)
        denom3 = (fc - fa) * (fc - fb)
        
        if abs(denom1) > 1e-15 and abs(denom2) > 1e-15 and abs(denom3) > 1e-15:
            try:
                term1 = (a * a) * fb * fc / denom1
                term2 = (b * b) * fa * fc / denom2
                term3 = (c * c) * fa * fb / denom3
                x_new = term1 + term2 + term3
                
                # Validate x_new
                if math.isfinite(x_new):
                    if a < b:
                        if not (a < x_new < b):
                            x_new = None
                    else:
                        if not (b < x_new < a):
                            x_new = None
            except (ZeroDivisionError, OverflowError):
                x_new = None

        if x_new is None:
            # Bisection fallback
            x_new = (a + b) / 2.0

        fx_new = f(x_new)
        if abs(fx_new) <= 1e-8:
            return x_new

        # Update Bracket
        if fa * fx_new < 0:
            b = x_new
            fb = fx_new
        else:
            a = x_new
            fa = fx_new
            
        # Update c to the new point
        c = x_new
        fc = fx_new

        # Convergence check
        if abs(b - a) < 1e-12:
            return a if abs(fa) < abs(fb) else b

    return c
# EVOLVE-BLOCK-END
