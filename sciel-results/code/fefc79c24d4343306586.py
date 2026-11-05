"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isnan(val) or math.isinf(val):
            return None
        a = abs(val)
        if a < best_abs:
            best_abs = a
            best_x = x
        return val
    
    def is_good(x):
        if not (math.isfinite(x)):
            return False
        fx = call_f(x)
        if fx is None:
            return False
        return abs(fx) <= TOL
    
    def newton_raphson(x0_val, max_iter=100):
        nonlocal evals, best_x, best_abs
        x = float(x0_val)
        for i in range(max_iter):
            if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
                break
            fx = call_f(x)
            if fx is None:
                break
            if abs(fx) <= TOL:
                return x
            h = 1e-7 * max(1.0, abs(x))
            fx_h = call_f(x + h)
            if fx_h is None:
                break
            d = (fx_h - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + math.copysign(10 * max(1.0, abs(x)), x_new - x)
            x = x_new
        return x
    
    def secant(x0_val, max_iter=100):
        nonlocal evals, best_x, best_abs
        x0 = float(x0_val)
        x1 = x0 + 1e-4 * max(1.0, abs(x0))
        fx0 = call_f(x0)
        if fx0 is None:
            return x0
        if abs(fx0) <= TOL:
            return x0
        fx1 = call_f(x1)
        if fx1 is None:
            return x0
        if abs(fx1) <= TOL:
            return x1
        for i in range(max_iter):
            if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
                break
            if fx1 == fx0:
                break
            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            if not math.isfinite(x2):
                break
            # Damping
            if abs(x2 - x1) > 10 * max(1.0, abs(x1)):
                x2 = x1 + math.copysign(10 * max(1.0, abs(x1)), x2 - x1)
            x0, x1 = x1, x2
            fx0 = fx1
            fx1 = call_f(x1)
            if fx1 is None:
                break
            if abs(fx1) <= TOL:
                return x1
        return x1
    
    # Try Newton first
    x_res = newton_raphson(x0)
    if is_good(x_res):
        return x_res
    
    # Try Secant
    x_res = secant(x0)
    if is_good(x_res):
        return x_res
    
    # Try Newton from a slightly perturbed start
    x_pert = x0 + 1e-3 * max(1.0, abs(x0))
    x_res = newton_raphson(x_pert)
    if is_good(x_res):
        return x_res
    
    # Try Secant from perturbed start
    x_res = secant(x_pert)
    if is_good(x_res):
        return x_res
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
