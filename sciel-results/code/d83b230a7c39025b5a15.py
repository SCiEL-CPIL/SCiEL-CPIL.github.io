"""
Robust scalar root finder for oscillatory equations.
Strategy:
1. Walk outward from x0 with midpoint checks to find a sign-change bracket.
2. Use Illinois/False-position method to converge to a root within the bracket.
3. Strict call budgeting and time guard.
"""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    MAX_CALLS = 490     # Stay under 500
    
    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')
    
    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        if calls >= MAX_CALLS:
            raise RuntimeError("Call limit exceeded")
        calls += 1
        val = f(x)
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = x
        return val
    
    # Initial evaluation
    try:
        f0 = _f(x0)
    except Exception:
        return best_x
    
    if abs(f0) <= 1e-8:
        return x0
    
    # Walk outward to find a bracket [a, b] with sign change
    # Use walking with midpoint checks for robustness to fast oscillations
    a = float(x0)
    fa = f0
    step = 1.0
    max_walk_steps = 200  # Limit walking steps to avoid excessive calls
    
    bracket_found = False
    b = None
    fb = None
    
    for direction in [1, -1]:
        if bracket_found:
            break
        step = 1.0
        for _ in range(max_walk_steps):
            if time.time() - start_time > TIME_LIMIT:
                break
            if calls >= MAX_CALLS:
                break
            x_new = x0 + direction * step
            try:
                f_new = _f(x_new)
            except Exception:
                break
            
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Check midpoint for sign change (robust to fast oscillations)
            mid = (x0 + direction * step / 2)
            # We already have f at x0 and x_new, check if sign changed
            if fa * f_new < 0:
                # Sign change between x0 and x_new
                # But we should check if there's a closer bracket
                # Use the bracket [x0, x_new] but refine by checking midpoint
                a = x0
                b = x_new
                fa = f0
                fb = f_new
                bracket_found = True
                break
            
            # Also check midpoint to catch fast oscillations
            if calls < MAX_CALLS - 1:
                try:
                    f_mid = _f(mid)
                except Exception:
                    f_mid = None
                
                if f_mid is not None:
                    if abs(f_mid) <= 1e-8:
                        return mid
                    # Check sign change between x0 and mid
                    if fa * f_mid < 0:
                        a = x0
                        b = mid
                        fa = f0
                        fb = f_mid
                        bracket_found = True
                        break
                    # Check sign change between mid and x_new
                    if f_mid * f_new < 0:
                        a = mid
                        b = x_new
                        fa = f_mid
                        fb = f_new
                        bracket_found = True
                        break
            
            # Update for next step
            x0 = x_new
            fa = f_new
            step *= 1.5  # Geometric expansion
            
            if abs(step) > 1e6:
                break
    
    if not bracket_found:
        # No bracket found; return best point seen
        if best_abs_f <= 1e-8:
            return best_x
        # Try a few more evaluations near x0 with smaller steps
        for dx in [0.1, -0.1, 0.01, -0.01, 0.5, -0.5]:
            if time.time() - start_time > TIME_LIMIT:
                break
            if calls >= MAX_CALLS:
                break
            x_try = float(x0) + dx
            try:
                f_try = _f(x_try)
            except Exception:
                continue
            if abs(f_try) <= 1e-8:
                return x_try
    
    if bracket_found:
        # Use Constrained Secant method with Bisection fallback
        # This is similar to Brent's method but simpler
        a, b = float(a), float(b)
        fa, fb = float(fa), float(fb)
        
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            # Shouldn't happen, but fallback to bisection
            pass
        
        for _ in range(100):
            if time.time() - start_time > TIME_LIMIT:
                break
            if calls >= MAX_CALLS:
                break
            
            # Secant step
            if fb - fa != 0:
                c = b - fb * (b - a) / (fb - fa)
            else:
                c = (a + b) / 2
            
            # If Secant step is outside bracket, use Bisection
            if c <= a or c >= b:
                c = (a + b) / 2
            
            try:
                fc = _f(c)
            except Exception:
                break
            
            if abs(fc) <= 1e-8:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            if abs(b - a) < 1e-12:
                break
    
    # Final check
    if best_abs_f <= 1e-8:
        return best_x
    
    # Return best point found
    return best_x

# EVOLVE-BLOCK-END
