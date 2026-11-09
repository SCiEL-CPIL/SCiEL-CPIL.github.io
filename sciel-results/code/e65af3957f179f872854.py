"""Direct root finder for smooth polynomials using Inverse Quadratic Interpolation and Secant fallback."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 145.0  # leave a margin before 150s kill
_SAFETY = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    tol = 1e-8
    max_calls = 480
    calls = 0

    def ceval(x):
        """Counted, safe evaluation."""
        nonlocal calls
        if calls >= max_calls:
            return None, False
        calls += 1
        try:
            val = f(x)
        except Exception:
            return None, False
        if not _is_finite(val):
            return None, False
        return val, True

    # --- Phase 0: Initial evaluation ---
    x = float(x0)
    fx, ok = ceval(x)
    if not ok:
        # Probe grid if x0 fails
        for dx in (1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 10.0, -10.0):
            xt = x + dx
            fx, ok = ceval(xt)
            if ok:
                x = xt
                break
        if not ok:
            return x

    if abs(fx) <= tol:
        return x

    # --- Phase 1: Get a second point ---
    h0 = 1e-4 * max(1.0, abs(x))
    x_prev = x
    fx_prev = fx
    x_curr = x + h0
    fx_curr, ok = ceval(x_curr)
    if not ok:
        x_curr = x - h0
        fx_curr, ok = ceval(x_curr)
    if not ok:
        return x

    if abs(fx_curr) <= tol:
        return x_curr

    # Track best point
    best_x = x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
    best_f = min(abs(fx_curr), abs(fx_prev))

    # We have 2 points. Let's try to get a 3rd to enable Quadratic Interpolation.
    # Use Secant to get the 3rd point.
    denom = fx_curr - fx_prev
    if abs(denom) < 1e-18:
        # Degenerate, nudge
        x_curr += 1e-6 * max(1.0, abs(x_curr))
        fx_curr, ok = ceval(x_curr)
        if not ok:
            return best_x
        if abs(fx_curr) <= tol:
            return x_curr
        denom = fx_curr - fx_prev

    x_next = x_curr - fx_curr * (x_curr - x_prev) / denom
    if not _is_finite(x_next):
        return best_x
        
    fx_next, ok = ceval(x_next)
    if not ok:
        return best_x

    if abs(fx_next) <= tol:
        return x_next

    # Update best
    if abs(fx_next) < best_f:
        best_f = abs(fx_next)
        best_x = x_next

    # Now we have 3 points: (x_prev, fx_prev), (x_curr, fx_curr), (x_next, fx_next)
    # Let's rename for clarity: p1, p2, p3
    x1, f1 = x_prev, fx_prev
    x2, f2 = x_curr, fx_curr
    x3, f3 = x_next, fx_next

    # --- Phase 2: Inverse Quadratic Interpolation (IQI) ---
    # Fit a quadratic x = A*f^2 + B*f + C passing through (f1, x1), (f2, x2), (f3, x3)
    # We want to find f such that x(f) is the root? No, we want x such that f(x)=0.
    # Inverse interpolation fits x as a function of f.
    # x(f) = a*f^2 + b*f + c
    # We solve for x(0).
    
    # Coefficients for quadratic interpolation of x vs f
    # x1 = a*f1^2 + b*f1 + c
    # x2 = a*f2^2 + b*f2 + c
    # x3 = a*f3^2 + b*f3 + c
    
    # Using Lagrange basis or solving linear system.
    # Let's use the formula for x(0) directly:
    # x(0) = x1 * (f2*f3) / ((f2-f1)*(f3-f1)) + x2 * (f1*f3) / ((f1-f2)*(f3-f2)) + x3 * (f1*f2) / ((f1-f3)*(f2-f3))
    # Wait, the standard Lagrange formula for x(0) given points (f1, x1)...
    # L1(0) = (0-f2)(0-f3) / ((f1-f2)(f1-f3)) = f2*f3 / ((f1-f2)(f1-f3))
    # L2(0) = (0-f1)(0-f3) / ((f2-f1)(f2-f3)) = f1*f3 / ((f2-f1)(f2-f3))
    # L3(0) = (0-f1)(0-f2) / ((f3-f1)(f3-f2)) = f1*f2 / ((f3-f1)(f3-f2))
    
    # x_new = x1*L1 + x2*L2 + x3*L3

    max_iter = 50
    for _ in range(max_iter):
        if calls >= max_calls or _time_left() < 0:
            break
            
        # Calculate IQI step
        # Check for division by zero or near-zero denominators
        d1 = f1 - f2
        d2 = f1 - f3
        d3 = f2 - f3
        
        use_iqi = True
        x_new = None
        
        if abs(d1) > 1e-15 and abs(d2) > 1e-15 and abs(d3) > 1e-15:
            L1 = (f2 * f3) / (d1 * d2)
            L2 = (f1 * f3) / (-d1 * d3) # (f2-f1)=-d1, (f2-f3)=d3
            L3 = (f1 * f2) / (d2 * -d3) # (f3-f1)=d2, (f3-f2)=-d3
            
            x_new = x1 * L1 + x2 * L2 + x3 * L3
            
            # Safeguard: If x_new is not in the bracket defined by the closest two points,
            # or is too far, fall back to Secant.
            # For polynomials, IQI is usually safe if points are close.
            # Let's check if it's "reasonable".
            if not _is_finite(x_new):
                use_iqi = False
            else:
                # Simple sanity check: distance from current best
                if abs(x_new - best_x) > 1e6 * max(1.0, abs(best_x)):
                    use_iqi = False

        if not use_iqi:
            # Fall back to Secant using the two most recent points (x3, f3) and (x2, f2)
            # Or the two with smallest |f|?
            # Secant is robust.
            if abs(f3 - f2) < 1e-18:
                # Degenerate, try x3 and x1
                if abs(f3 - f1) < 1e-18:
                    break
                x_new = x3 - f3 * (x3 - x1) / (f3 - f1)
            else:
                x_new = x3 - f3 * (x3 - x2) / (f3 - f2)
                
            if not _is_finite(x_new):
                break

        # Evaluate new point
        f_new, ok = ceval(x_new)
        if not ok:
            break

        if abs(f_new) <= tol:
            return x_new

        if abs(f_new) < best_f:
            best_f = abs(f_new)
            best_x = x_new

        # Update points for next iteration
        # Keep the 3 "best" or most recent?
        # Usually, we keep the 3 most recent to maintain locality.
        x1, f1 = x2, f2
        x2, f2 = x3, f3
        x3, f3 = x_new, f_new

    return best_x
