"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, using a portfolio of robust methods.
    Uses a deadline guard to stay within the wall-clock budget.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
    tol = 1e-10  # target |f(x)| <= 1e-8, use tighter internal tol
    
    x = float(x0)
    best_x = x
    best_f = f(x)
    best_abs = abs(best_f) if best_f is not None and math.isfinite(best_f) else float('inf')
    
    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if x_val is None or f_val is None:
            return
        if not math.isfinite(x_val) or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f
    
    # Try multiple strategies, each with a call budget
    # Strategy 1: Secant method (fast, 2 calls per iteration)
    def secant_method(x0, max_calls=400):
        if time.time() > deadline:
            return
        x1 = x0
        x2 = x0 + 1e-4 * max(1.0, abs(x0))
        f1 = f(x1)
        f2 = f(x2)
        update_best(x1, f1)
        update_best(x2, f2)
        calls = 2
        
        for _ in range(200):
            if time.time() > deadline:
                return
            if abs(f2) <= tol:
                return
            if abs(f1) <= tol:
                return
            denom = f2 - f1
            if denom == 0:
                break
            x3 = x1 - f1 * (x2 - x1) / denom
            f3 = f(x3)
            calls += 1
            update_best(x3, f3)
            if abs(f3) <= tol:
                return
            if not math.isfinite(x3) or not math.isfinite(f3):
                break
            x1, f1 = x2, f2
            x2, f2 = x3, f3
        
    # Strategy 2: Bisection with expanding bracket
    def bisection_method(x0, max_calls=400):
        if time.time() > deadline:
            return
        # Try to find a bracket around x0
        h = 1e-4 * max(1.0, abs(x0))
        for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            if time.time() > deadline:
                return
            a = x0 - h * scale
            b = x0 + h * scale
            fa = f(a)
            fb = f(b)
            update_best(a, fa)
            update_best(b, fb)
            if not math.isfinite(fa) or not math.isfinite(fb):
                continue
            if fa * fb <= 0:
                # Found a bracket, do bisection
                for _ in range(60):
                    if time.time() > deadline:
                        return
                    m = (a + b) / 2.0
                    fm = f(m)
                    update_best(m, fm)
                    if not math.isfinite(fm):
                        break
                    if abs(fm) <= tol:
                        return
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                return
        # If no bracket found, try Newton with finite diff
        newton_method(x0, max_calls)
    
    # Strategy 3: Newton's method with finite-difference derivative
    def newton_method(x0, max_calls=400):
        if time.time() > deadline:
            return
        x = float(x0)
        for _ in range(100):
            if time.time() > deadline:
                return
            fx = f(x)
            update_best(x, fx)
            if abs(fx) <= tol:
                return
            if not math.isfinite(fx):
                break
            h = 1e-8 * max(1.0, abs(x))
            fxh = f(x + h)
            if not math.isfinite(fxh):
                break
            d = (fxh - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            x = x_new
    
    # Run strategies in order of expected efficiency
    secant_method(x0)
    if best_abs > tol:
        bisection_method(x0)
    if best_abs > tol:
        newton_method(x0)
    
    # Final fallback: if we still haven't converged, return best found
    if best_abs <= 1e-8:
        return best_x
    
    # If nothing worked, try a few more random-ish perturbations
    for delta in [1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0]:
        if time.time() > deadline:
            break
        x_try = x0 + delta * max(1.0, abs(x0))
        f_try = f(x_try)
        update_best(x_try, f_try)
        if best_abs <= tol:
            return best_x
    
    return best_x
# EVOLVE-BLOCK-END
