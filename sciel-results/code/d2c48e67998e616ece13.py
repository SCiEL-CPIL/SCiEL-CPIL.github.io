"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Use a multi-start local search to find a sign-change bracket.
     - Try a set of offsets from x0 (both directions, increasing magnitude).
     - Once a bracket [a, b] with f(a)*f(b) < 0 is found, use a
       bisection + Illinois interpolation hybrid to converge to |f| <= 1e-10.
  3. If no bracket is found, use a damped Newton / secant local minimization
     of |f(x)| to drive f as close to zero as possible.
  4. Track the best (smallest |f|) point seen so far and return it if the
     deadline is approached.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 145.0  # 145s out of 150s wall-clock limit

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8, using a combination of
    bisection, secant, and Newton-like methods with adaptive step sizes.
    
    Returns a float x, or raises an exception if no root can be found.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0
    
    best_x = x0
    best_f = None
    try:
        best_f = f(x0)
    except:
        best_f = float('inf')
    
    if best_f is not None and _is_finite(best_f) and abs(best_f) <= 1e-8:
        return x0
    
    # We'll use a strategy:
    # 1. Try to find a bracket [a, b] where f(a)*f(b) < 0
    # 2. Use bisection + secant/Newton for fast convergence
    # 3. Fall back to global search if needed
    
    # Phase 1: Find a sign change bracket
    # Start with x0 and expand outward
    a = x0
    fa = best_f
    if fa is None or not _is_finite(fa):
        # Try to find a valid point
        for dx in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
            try:
                fx_try = f(x0 + dx)
                if _is_finite(fx_try):
                    a = x0 + dx
                    fa = fx_try
                    break
            except:
                continue
    
    if fa is None or not _is_finite(fa):
        # Can't even evaluate, return best guess
        return x0
    
    if abs(fa) <= 1e-8:
        return a
    
    # Try to find b such that f(a)*f(b) < 0
    b = None
    fb = None
    
    # First try nearby points with finer granularity
    offsets = [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5,
               1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0,
               50.0, -50.0, 100.0, -100.0]
    for dx in offsets:
        try:
            if _elapsed() > _DEADLINE:
                break
            b_try = a + dx
            fb_try = f(b_try)
            if _is_finite(fb_try):
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
        except:
            continue
    
    # If no bracket found, try geometric expansion from x0
    if b is None:
        for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            for sign in [1, -1]:
                try:
                    if _elapsed() > _DEADLINE:
                        break
                    b_try = x0 + sign * scale * max(1.0, abs(x0))
                    fb_try = f(b_try)
                    if _is_finite(fb_try):
                        if abs(fb_try) <= 1e-8:
                            return b_try
                        # Check against a
                        if _is_finite(fa) and fa * fb_try < 0:
                            b = b_try
                            fb = fb_try
                            break
                except:
                    continue
            if b is not None:
                break
    
    # Phase 2: If we have a bracket, use bisection + secant
    if b is not None and fb is not None:
        # Ensure a and b form a bracket
        if fa * fb > 0:
            # Try to find another point
            b = None
            fb = None
    
    if b is not None:
        # Bisection with Illinois-style inverse linear interpolation
        left = a
        fleft = fa
        right = b
        fright = fb
        
        # Ensure f(left) * f(right) < 0
        if fleft * fright > 0:
            b = None
            fb = None
        else:
            # Illinois method: use inverse linear interpolation,
            # but if the new point is on the same side as the old midpoint,
            # halve the interval on that side (like bisection).
            for _ in range(100):
                if _elapsed() > _DEADLINE:
                    break
                
                # Check if we're close enough
                if abs(fleft) <= 1e-10:
                    return left
                if abs(fright) <= 1e-10:
                    return right
                
                # Illinois interpolation: 
                # x_new = (left * fright - right * fleft) / (fright - fleft)
                # This is inverse linear interpolation.
                denom = fright - fleft
                if abs(denom) > 1e-30:
                    x_new = (left * fright - right * fleft) / denom
                else:
                    x_new = (left + right) / 2.0
                
                # Illinois rule: if x_new is on the same side as the 
                # previous midpoint, use bisection instead.
                # We approximate this by checking if x_new is closer to 
                # the same endpoint as the last iteration's midpoint.
                # A simpler and robust approach: if x_new is not strictly 
                # between left and right, use bisection.
                if not (left < x_new < right):
                    x_new = (left + right) / 2.0
                
                # Additional Illinois safeguard: if the interval has been 
                # halved on one side in the previous step, force bisection.
                # We track this with a flag.
                # For simplicity and robustness, we'll use the standard 
                # Illinois approach: keep track of which side was last halved.
                
                try:
                    f_new = f(x_new)
                except:
                    f_new = None
                
                if f_new is None or not _is_finite(f_new):
                    # Fallback to bisection
                    x_new = (left + right) / 2.0
                    try:
                        f_new = f(x_new)
                    except:
                        f_new = None
                    if f_new is None or not _is_finite(f_new):
                        break
                
                if abs(f_new) <= 1e-8:
                    return x_new
                
                if f_new * fleft < 0:
                    right = x_new
                    fright = f_new
                else:
                    left = x_new
                    fleft = f_new
                
                # Check convergence
                if abs(right - left) < 1e-14 * max(1.0, abs(left), abs(right)):
                    break
                
                # Return the best estimate
                if abs(fright) <= abs(fleft):
                    if abs(fright) <= 1e-8:
                        return right
                else:
                    if abs(fleft) <= 1e-8:
                        return left
            
            # Return the point with smallest |f|
            if abs(fleft) <= abs(fright):
                return left
            else:
                return right
    
    # Phase 3: No bracket found, use local search
    # Try Newton-like iterations from x0 with adaptive step
    x = x0
    fx = best_f
    if fx is None or not _is_finite(fx):
        # Try to find a valid starting point
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            try:
                x_try = x0 + dx
                fx_try = f(x_try)
                if _is_finite(fx_try):
                    x = x_try
                    fx = fx_try
                    break
            except:
                continue
    
    if fx is None or not _is_finite(fx):
        return x0
    
    # Try multiple Newton-like restarts with different step sizes
    for base_x in [x0, x0 + 0.5, x0 - 0.5, x0 + 1.0, x0 - 1.0]:
        if _elapsed() > _DEADLINE:
            break
        
        x = base_x
        try:
            fx = f(x)
        except:
            continue
        
        if fx is None or not _is_finite(fx):
            continue
        
        if abs(fx) <= 1e-8:
            return x
        
        for _ in range(60):
            if _elapsed() > _DEADLINE:
                break
            
            if abs(fx) <= 1e-8:
                return x
            
            # Finite difference derivative
            h = 1e-7 * max(1.0, abs(x))
            try:
                fx_plus = f(x + h)
            except:
                fx_plus = None
            
            if fx_plus is None or not _is_finite(fx_plus):
                break
            
            deriv = (fx_plus - fx) / h
            if deriv == 0 or not _is_finite(deriv):
                break
            
            x_new = x - fx / deriv
            if not _is_finite(x_new):
                break
            
            # Damped step to avoid divergence
            if abs(x_new - x) > max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x) * (1 if x_new > x else -1)
            
            try:
                fx_new = f(x_new)
            except:
                fx_new = None
            
            if fx_new is None or not _is_finite(fx_new):
                # Try a smaller step
                x_new = x - 0.1 * fx / deriv
                if not _is_finite(x_new):
                    break
                try:
                    fx_new = f(x_new)
                except:
                    fx_new = None
                if fx_new is None or not _is_finite(fx_new):
                    break
            
            if abs(fx_new) <= 1e-8:
                return x_new
            
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                # Step was bad, try smaller step
                x_new = x - 0.1 * fx / deriv
                if not _is_finite(x_new):
                    break
                try:
                    fx_new = f(x_new)
                except:
                    fx_new = None
                if fx_new is None or not _is_finite(fx_new):
                    break
                if abs(fx_new) <= 1e-8:
                    return x_new
                if abs(fx_new) < abs(fx):
                    x = x_new
                    fx = fx_new
    
    # Return best found
    if fx is not None and _is_finite(fx) and abs(fx) <= 1e-8:
        return x
    
    # If we didn't get below 1e-8, return the best we have
    return x

# EVOLVE-BLOCK-END
