"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a secant-based method with
    adaptive bracketing and a final Newton polish.
    
    Strategy:
    1. Use secant method (1 new eval per iteration) for fast convergence.
    2. Track best (smallest |f|) point seen.
    3. If secant diverges or stalls, fall back to bisection on a bracket.
    4. Final polish with a few Newton steps using finite-diff derivative.
    """
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    best_x = x
    best_fx = fx
    best_abs = abs(fx)
    
    if best_abs <= 1e-8:
        return x
    
    # Secant method setup
    # Choose a second point
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    fx1 = f(x1)
    
    if abs(fx1) < best_abs:
        best_x = x1
        best_fx = fx1
        best_abs = abs(fx1)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Track the two most recent points for secant
    x_prev = x
    fx_prev = fx
    x_curr = x1
    fx_curr = fx1
    
    # Also track a bracket if we find one
    lo = None
    hi = None
    lo_fx = None
    hi_fx = None
    
    max_iter = 40
    for i in range(max_iter):
        if _time_left() < 1.0:
            break
        
        # Check for sign change to establish bracket
        if fx_prev * fx_curr < 0:
            if fx_prev < 0:
                lo, hi = x_prev, x_curr
                lo_fx, hi_fx = fx_prev, fx_curr
            else:
                lo, hi = x_curr, x_prev
                lo_fx, hi_fx = fx_curr, fx_prev
        
        # Secant step
        if fx_curr - fx_prev == 0:
            # Degenerate: try a small perturbation
            x_new = x_curr + 1e-6 * max(1.0, abs(x_curr))
        else:
            x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        # Ensure x_new is finite
        if not _is_finite(x_new):
            # Fallback: bisection if we have a bracket
            if lo is not None and hi is not None:
                x_new = 0.5 * (lo + hi)
            else:
                x_new = x_curr + 1e-4 * max(1.0, abs(x_curr))
        
        # Limit step size to prevent wild jumps
        step = abs(x_new - x_curr)
        max_step = 10.0 * max(1.0, abs(x_curr))
        if step > max_step:
            x_new = x_curr + math.copysign(max_step, x_new - x_curr)
        
        fx_new = f(x_new)
        
        if not _is_finite(fx_new):
            # Bad point, try bisection or small step
            if lo is not None and hi is not None:
                x_new = 0.5 * (lo + hi)
                fx_new = f(x_new)
                if not _is_finite(fx_new):
                    break
            else:
                x_new = x_curr + 1e-5 * max(1.0, abs(x_curr))
                fx_new = f(x_new)
                if not _is_finite(fx_new):
                    break
        
        abs_new = abs(fx_new)
        if abs_new < best_abs:
            best_x = x_new
            best_fx = fx_new
            best_abs = abs_new
        
        if best_abs <= 1e-8:
            return best_x
        
        # Update bracket
        if fx_new * fx_curr < 0:
            if fx_new < 0:
                lo, hi = x_new, x_curr
                lo_fx, hi_fx = fx_new, fx_curr
            else:
                lo, hi = x_curr, x_new
                lo_fx, hi_fx = fx_curr, fx_new
        elif fx_new * fx_prev < 0:
            if fx_new < 0:
                lo, hi = x_new, x_prev
                lo_fx, hi_fx = fx_new, fx_prev
            else:
                lo, hi = x_prev, x_new
                lo_fx, hi_fx = fx_prev, fx_new
        
        # If we have a bracket and secant is not making progress, use bisection
        if lo is not None and hi is not None:
            bracket_width = abs(hi - lo)
            # If secant step doesn't reduce bracket width, bisect
            if abs(x_new - x_curr) > bracket_width * 0.5 or i > 5:
                # Use bisection for reliability
                x_new = 0.5 * (lo + hi)
                fx_new = f(x_new)
                if not _is_finite(fx_new):
                    break
                abs_new = abs(fx_new)
                if abs_new < best_abs:
                    best_x = x_new
                    best_fx = fx_new
                    best_abs = abs_new
                if best_abs <= 1e-8:
                    return best_x
                # Update bracket
                if fx_new * lo_fx < 0:
                    hi = x_new
                    hi_fx = fx_new
                else:
                    lo = x_new
                    lo_fx = fx_new
                # Continue with secant from the two bracket endpoints
                x_prev = lo
                fx_prev = lo_fx
                x_curr = hi
                fx_curr = hi_fx
                continue
        
        # Update secant points
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_new
        fx_curr = fx_new
    
    # Final polish: a few Newton steps from best_x
    x = best_x
    fx = best_fx
    for _ in range(10):
        if _time_left() < 0.5:
            break
        if abs(fx) <= 1e-8:
            break
        h = 1e-7 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0 or not _is_finite(d):
            break
        x_new = x - fx / d
        if not _is_finite(x_new):
            break
        fx_new = f(x_new)
        if not _is_finite(fx_new):
            break
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            break
    
    if abs(fx) < best_abs:
        return x
    return best_x
