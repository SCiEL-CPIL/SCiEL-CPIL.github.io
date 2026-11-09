"""Root finder for 3 fixed oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 110.0  # safety margin before 120s limit
    x = float(x0)
    
    # Try to evaluate f at x0
    try:
        fx = f(x)
    except Exception:
        return x
    
    if not math.isfinite(fx):
        return x
    
    if abs(fx) <= 1e-8:
        return x
    
    # Strategy: find a bracket [a, b] where f(a) and f(b) have opposite signs,
    # then use a hybrid method (Brent-like with secant acceleration) to converge.
    
    # First, try to bracket the root by expanding outward from x0.
    # We'll search in both directions.
    
    a = None
    b = None
    fa = None
    fb = None
    
    # Track best candidate so far (smallest |f|)
    best_x = x
    best_fx = fx
    best_abs = abs(fx)
    
    # Adaptive expansion: start with small step, double each time
    base_step = 0.5 * max(1.0, abs(x))
    step = base_step
    max_step = 1e8 * max(1.0, abs(x))
    
    # Search outward in both directions
    found_bracket = False
    
    # Try positive direction first
    for _ in range(60):
        if time.time() > deadline:
            break
        x_right = x + step
        try:
            f_right = f(x_right)
        except Exception:
            break
        if not math.isfinite(f_right):
            break
        if abs(f_right) < best_abs:
            best_x = x_right
            best_fx = f_right
            best_abs = abs(f_right)
        if abs(f_right) <= 1e-8:
            return x_right
        if fa is None:
            fa = fx
            a = x
        if fa * f_right < 0:
            a = x
            fa = fx
            b = x_right
            fb = f_right
            found_bracket = True
            break
        # If we've gone too far, try the other direction
        step *= 2
        if step > max_step:
            break
    
    # If no bracket found going right, try going left
    if not found_bracket:
        step = base_step
        for _ in range(60):
            if time.time() > deadline:
                break
            x_left = x - step
            try:
                f_left = f(x_left)
            except Exception:
                break
            if not math.isfinite(f_left):
                break
            if abs(f_left) < best_abs:
                best_x = x_left
                best_fx = f_left
                best_abs = abs(f_left)
            if abs(f_left) <= 1e-8:
                return x_left
            if fa is None:
                fa = fx
                a = x
            if fa * f_left < 0:
                a = x_left
                fa = f_left
                b = x
                fb = fx
                found_bracket = True
                break
            step *= 2
            if step > max_step:
                break
    
    # If we found a bracket, use a hybrid root-finding method
    if found_bracket:
        result = _hybrid_brent(f, a, b, fa, fb, deadline)
        if result is not None:
            return result
    
    # If no bracket found, try secant method from x0
    if time.time() < deadline:
        result = _secant(f, x, fx, deadline, best_x, best_fx)
        if result is not None:
            return result
    
    # Return best candidate found
    return best_x


def _hybrid_brent(f, a, b, fa, fb, deadline):
    """Brent-like method with secant acceleration."""
    max_iter = 200
    tol = 1e-12
    xtol = 1e-14
    
    # Ensure fa * fb < 0
    if fa * fb > 0:
        return None
    
    # Initial guess: use the point closer to zero
    c = a
    fc = fa
    d = b
    fd = fb
    e = 0.0  # previous step
    mflag = True
    
    # Use the point with smaller |f| as initial x
    if abs(fa) < abs(fb):
        x = a
        fx = fa
    else:
        x = b
        fx = fb
    
    for _ in range(max_iter):
        if time.time() > deadline:
            if abs(fx) <= 1e-8:
                return x
            break
        if abs(fx) <= 1e-8:
            return x
        if abs(b - a) < xtol:
            return (a + b) / 2.0
        
        # Update bracket
        if fa * fc < 0:
            b = a
            fb = fa
            a = c
            fa = fc
            e = b - a
            mflag = True
        elif fb * fc < 0:
            a = c
            fa = fc
        else:
            if abs(fb) < abs(fa):
                a = b
                fa = fb
            e = b - a  # previous step
        
        # Choose new point
        if mflag:
            # Secant step
            if a != b:
                mstep = (a - b) * fa / (fa - fb) if fa != fb else e / 2.0
            else:
                mstep = e / 2.0
            # Bisection if secant step is too large
            if not (0 < mstep < abs(e) / 2) or (e == 0):
                mstep = e / 2.0
                mflag = False
            c = a + mstep
            fc = f(c)
        else:
            # Inverse quadratic interpolation
            if a != b and b != c:
                # Compute Lagrange interpolation
                t1 = (a - b) * fa / (fa - fb) if fa != fb else 0
                t2 = (a - c) * fa / (fa - fc) if fa != fc else 0
                t3 = (b - c) * fb / (fb - fc) if fb != fc else 0
                mstep = t1 + t2 + t3
            else:
                mstep = e / 2.0
                mflag = True
            # Bisection if interpolation step is too large
            if not (0 < mstep < abs(e) / 2) or (e == 0):
                mstep = e / 2.0
                mflag = True
            c = a + mstep
            fc = f(c)
        
        # Update best
        if abs(fc) < abs(fx):
            x = c
            fx = fc
        
        # Check convergence
        if abs(c - a) < xtol:
            return c
        if abs(fc) <= 1e-8:
            return c
    
    # Return best
    if abs(fx) <= 1e-8:
        return x
    return x


def _secant(f, x0, fx0, deadline, best_x, best_fx):
    """Secant method fallback."""
    x = x0
    fx = fx0
    h = 1e-6 * max(1.0, abs(x))
    x1 = x + h
    try:
        fx1 = f(x1)
    except Exception:
        return best_x
    if not math.isfinite(fx1):
        return best_x
    
    for _ in range(100):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-8:
            return x
        if abs(fx1) <= 1e-8:
            return x1
        if fx == fx1:
            break
        x2 = x1 - fx1 * (x1 - x) / (fx1 - fx)
        x, x1 = x1, x2
        fx, fx1 = fx1, f(x1)
        if not math.isfinite(fx1):
            break
        if abs(fx1) < abs(best_fx):
            best_x = x1
            best_fx = fx1
    
    if abs(fx) <= 1e-8:
        return x
    if abs(fx1) <= 1e-8:
        return x1
    return best_x
# EVOLVE-BLOCK-END
