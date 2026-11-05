"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Strategy:
    1. Try a few quick Newton/secant steps with damping.
    2. If not converged, try to bracket the root and use bisection/ Brent's method.
    3. Use a deadline guard to stay within time limits.
    4. Track best solution found (min |f(x)|) and return it.
    """
    deadline = time.time() + 145  # Leave 5s margin before 150s kill
    max_evals = 480
    eval_count = [0]
    
    def eval_f(x):
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if not math.isfinite(val):
                return None
            return val
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
            eval_count[0] += 1
            return None
    
    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return x
    best_x = x
    best_fx = abs(fx)
    
    if best_fx <= 1e-8:
        return x
    
    # Phase 1: Newton with damping and secant fallback
    x_prev = None
    fx_prev = None
    for i in range(30):
        if time.time() > deadline:
            break
        if fx is None:
            break
        if abs(fx) < best_fx:
            best_fx = abs(fx)
            best_x = x
        if best_fx <= 1e-8:
            return x
        
        # Try Newton step with finite difference derivative
        h = 1e-7 * max(1.0, abs(x))
        fxh = eval_f(x + h)
        if fxh is None:
            break
        d = (fxh - fx) / h
        if abs(d) < 1e-15:
            # Derivative too small, try secant or break
            if x_prev is not None and fx_prev is not None:
                denom = fx - fx_prev
                if abs(denom) > 1e-15:
                    step = -fx * (x - x_prev) / denom
                    # Damp the step
                    if abs(step) > 10 * max(1.0, abs(x)):
                        step = math.copysign(10 * max(1.0, abs(x)), step)
                    x_new = x + step
                    fx_new = eval_f(x_new)
                    if fx_new is None:
                        break
                    if abs(fx_new) < best_fx:
                        best_fx = abs(fx_new)
                        best_x = x_new
                    if best_fx <= 1e-8:
                        return x_new
                    x_prev, fx_prev = x, fx
                    x, fx = x_new, fx_new
                else:
                    break
            else:
                break
        else:
            step = -fx / d
            # Damp the step to avoid divergence
            if abs(step) > 10 * max(1.0, abs(x)):
                step = math.copysign(10 * max(1.0, abs(x)), step)
            x_new = x + step
            fx_new = eval_f(x_new)
            if fx_new is None:
                # Try smaller step
                step = step * 0.1
                x_new = x + step
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
            if abs(fx_new) < best_fx:
                best_fx = abs(fx_new)
                best_x = x_new
            if best_fx <= 1e-8:
                return x_new
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
    
    if best_fx <= 1e-8:
        return best_x
    
    # Phase 2: Try to bracket the root
    # Expand from x0 in both directions
    a = x0
    b = x0
    fa = eval_f(a)
    fb = eval_f(b)
    
    if fa is None or fb is None:
        return best_x
    
    # Try expanding the bracket
    scale = max(1.0, abs(x0))
    for i in range(20):
        if time.time() > deadline:
            break
        a = x0 - (i + 1) * scale
        b = x0 + (i + 1) * scale
        fa = eval_f(a)
        fb = eval_f(b)
        if fa is None or fb is None:
            break
        if fa * fb < 0:
            break
        if abs(fa) < best_fx:
            best_fx = abs(fa)
            best_x = a
        if abs(fb) < best_fx:
            best_fx = abs(fb)
            best_x = b
        if best_fx <= 1e-8:
            return best_x
    
    if fa is not None and fb is not None and fa * fb < 0:
        # Bisection method
        for i in range(100):
            if time.time() > deadline:
                break
            mid = (a + b) / 2
            fm = eval_f(mid)
            if fm is None:
                break
            if abs(fm) < best_fx:
                best_fx = abs(fm)
                best_x = mid
            if best_fx <= 1e-8:
                return mid
            if fm * fa < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
    
    return best_x
# EVOLVE-BLOCK-END
