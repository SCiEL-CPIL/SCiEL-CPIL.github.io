import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 495

def _taylor_approximate_root(f, x0, counter, max_iter=30):
    """
    Build a local Taylor polynomial of degree 4 using finite differences,
    then use Newton iterations on the polynomial to find root candidates.
    Only call f to verify candidates.
    """
    # Step size for finite differences
    h = 1e-4 * max(1.0, abs(x0))
    
    # Evaluate f at 5 points for 4th-order Taylor approximation
    # Points: x0-2h, x0-h, x0, x0+h, x0+2h
    points = [x0 - 2*h, x0 - h, x0, x0 + h, x0 + 2*h]
    vals = []
    for p in points:
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            return None, None, False
        v = _safe_call(f, p)
        counter[0] += 1
        if math.isnan(v):
            return None, None, False
        vals.append(v)
    
    # Compute derivatives using finite differences
    # f' ≈ (-f(x-2h) + 8f(x-h) - 8f(x+h) + f(x+2h)) / (12h)
    # f'' ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x+2h)) / (2h^2)
    # f''' ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x+2h)) / (2h^3) -- wait, this is wrong
    # Let me use standard formulas:
    # f'(x) ≈ (-f(x-2h) + 8f(x-h) - 8f(x+h) + f(x+2h)) / (12h)
    # f''(x) ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x-2h)) / (2h^2) -- need correct formula
    # Actually: f''(x) ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x+2h)) / (2h^2) is incorrect
    # Correct: f''(x) ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x+2h)) / (2h^2) -- let me use standard
    # Standard 5-point: f''(x) ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x+2h)) / (2h^2)
    # Wait, the correct 5-point second derivative is:
    # f''(x) ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x+2h)) / (2h^2) -- this doesn't look right
    # Let me use: f''(x) ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x+2h)) / (2*h**2)
    # Actually the standard 5-point central difference for second derivative is:
    # f''(x) ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x+2h)) / (2*h**2) -- I'm confusing myself
    # Let me just use: f''(x) ≈ (f(x-h) - 2*f(x) + f(x+h)) / h**2 for simplicity, or
    # Use the 5-point: f''(x) ≈ (f(x-2h) - 2f(x-h) + 2f(x+h) - f(x+2h)) / (2h^2)
    # I'll use simpler 3-point for derivatives to avoid confusion:
    
    # f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
    # f'''(x) ≈ (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (2h^3)
    # f''''(x) ≈ (f(x+2h) - 4f(x+h) + 6f(x) - 4f(x-h) + f(x-2h)) / h^4
    
    fx = vals[2]
    f1 = (vals[3] - vals[1]) / (2*h)
    f2 = (vals[3] - 2*fx + vals[1]) / (h*h)
    f3 = (vals[4] - 2*vals[3] + 2*vals[1] - vals[0]) / (2*h**3)
    f4 = (vals[4] - 4*vals[3] + 6*fx - 4*vals[1] + vals[0]) / (h**4)
    
    # Taylor polynomial: P(t) = fx + f1*t + f2*t^2/2 + f3*t^3/6 + f4*t^4/24
    # P'(t) = f1 + f2*t + f3*t^2/2 + f4*t^3/6
    
    def poly_val(t):
        return fx + f1*t + 0.5*f2*t*t + (f3/6.0)*t**3 + (f4/24.0)*t**4
    
    def poly_deriv(t):
        return f1 + f2*t + 0.5*f3*t*t + (f4/6.0)*t**3
    
    # Newton iteration on the polynomial
    t = 0.0
    for _ in range(max_iter):
        pv = poly_val(t)
        if abs(pv) <= 1e-12:
            break
        pd = poly_deriv(t)
        if abs(pd) < 1e-15:
            break
        t_new = t - pv / pd
        if abs(t_new - t) < 1e-14 * max(1.0, abs(t_new)):
            t = t_new
            break
        if abs(t_new) > 10*h:  # Stay in local region
            t = t_new
            break
        t = t_new
    
    x_candidate = x0 + t
    
    # Verify with actual function call
    if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
        return x_candidate, None, False
    
    fc = _safe_call(f, x_candidate)
    counter[0] += 1
    
    if math.isnan(fc):
        return x_candidate, fc, False
    
    return x_candidate, fc, abs(fc) <= 1e-8

def _time_left():
    return _DEADLINE - time.time()

def _safe_call(f, x):
    try:
        val = f(float(x))
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        return float('nan')

def _newton_secant(f, x0, counter, max_iter=50):
    """
    Hybrid Newton-Secant method.
    Uses derivative approximation if possible, otherwise secant.
    Since we don't have f', we use secant primarily but with a safeguard.
    """
    x1 = x0
    x2 = x0 + 1e-6 * max(1.0, abs(x0))
    fx1 = _safe_call(f, x1)
    counter[0] += 1
    if math.isnan(fx1):
        return x1, fx1, False
    
    fx2 = _safe_call(f, x2)
    counter[0] += 1
    if math.isnan(fx2):
        return x2, fx2, False
    
    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if abs(fx2) <= 1e-8:
            return x2, fx2, True
        if abs(fx1) <= 1e-8:
            return x1, fx1, True
        
        # Secant step
        if fx2 == fx1:
            break
        x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
        if not math.isfinite(x_new):
            break
            
        x1, fx1 = x2, fx2
        x2 = x_new
        fx2 = _safe_call(f, x2)
        counter[0] += 1
        if math.isnan(fx2):
            break
            
    if abs(fx2) <= 1e-8:
        return x2, fx2, True
    if abs(fx1) <= 1e-8:
        return x1, fx1, True
    return x2, fx2, False

def _brent_like(f, a, b, fa, fb, counter):
    """
    Use scipy's brentq if available, otherwise fall back to bisection/secant.
    """
    if _HAS_SCIPY:
        def wrapped_f(x):
            if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                raise Exception("budget")
            val = _safe_call(f, x)
            counter[0] += 1
            if math.isnan(val):
                raise Exception("nan")
            return val
        
        try:
            x = _scipy_brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
            fx = _safe_call(f, x)
            counter[0] += 1
            if math.isnan(fx):
                return x, fx, False
            return x, fx, abs(fx) <= 1e-8
        except Exception:
            pass

    # Fallback: Bisection with Illinois acceleration
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    for _ in range(100):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break
            
        # Illinois method
        if abs(fa) < abs(fb):
            # Move b
            c = (a + b) / 2.0
            fc = _safe_call(f, c)
            counter[0] += 1
            if math.isnan(fc):
                break
            if abs(fc) <= 1e-8:
                return c, fc, True
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        else:
            # Move a
            c = (a + b) / 2.0
            fc = _safe_call(f, c)
            counter[0] += 1
            if math.isnan(fc):
                break
            if abs(fc) <= 1e-8:
                return c, fc, True
            if fb * fc < 0:
                a = c
                fa = fc
            else:
                b = c
                fb = fc
                
        if abs(b - a) < 1e-14 * max(1.0, abs(c)):
            break
            
    c = (a + b) / 2.0
    fc = _safe_call(f, c)
    counter[0] += 1
    return c, fc, abs(fc) <= 1e-8

def _find_bracket(f, x0, counter):
    """
    Find a bracket [a, b] such that f(a)*f(b) < 0.
    Uses exponential expansion from x0.
    """
    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if math.isnan(fx0):
        return None
    if abs(fx0) <= 1e-8:
        return (x0, x0, fx0, fx0)

    # Try expanding in both directions
    # Direction +1
    a = x0
    fa = fx0
    step = 1e-3
    for _ in range(50):
        if counter[0] >= _CALL_BUDGET - 10 or _time_left() < 1.0:
            break
        b = a + step
        fb = _safe_call(f, b)
        counter[0] += 1
        if math.isnan(fb):
            break
        if fa * fb < 0:
            return (a, b, fa, fb)
        a = b
        fa = fb
        step *= 2.0
        if step > 1e6:
            break

    # Direction -1
    a = x0
    fa = fx0
    step = 1e-3
    for _ in range(50):
        if counter[0] >= _CALL_BUDGET - 10 or _time_left() < 1.0:
            break
        b = a - step
        fb = _safe_call(f, b)
        counter[0] += 1
        if math.isnan(fb):
            break
        if fa * fb < 0:
            return (a, b, fa, fb)
        a = b
        fa = fb
        step *= 2.0
        if step > 1e6:
            break

    return None

def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Check x0
    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if not math.isnan(fx0) and abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0) if not math.isnan(fx0) else float('inf')

    # 2. Try Taylor approximation first - uses 6 calls total (5 for model + 1 verify)
    #    This can be very efficient if the local model is accurate
    if counter[0] < _CALL_BUDGET - 20 and _time_left() > 1.0:
        x, fx, success = _taylor_approximate_root(f, x0, counter)
        if fx is not None and not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x = x
                best_fx = fx
                best_abs = abs(fx)
            if success:
                return x

    # 3. Try local optimization (Secant) 
    if counter[0] < _CALL_BUDGET - 20 and _time_left() > 1.0:
        x, fx, success = _newton_secant(f, x0, counter)
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x = x
                best_fx = fx
                best_abs = abs(fx)
            if success:
                return x

    # 4. Try to find a bracket using exponential expansion
    bracket = _find_bracket(f, x0, counter)
    
    if bracket:
        a, b, fa, fb = bracket
        if a == b:
            return a
        x, fx, success = _brent_like(f, a, b, fa, fb, counter)
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x = x
                best_fx = fx
                best_abs = abs(fx)
            if success:
                return x

    # 5. If no bracket found or convergence failed, try Taylor from best point
    if counter[0] < _CALL_BUDGET - 15 and _time_left() > 1.0:
        x, fx, success = _taylor_approximate_root(f, best_x, counter)
        if fx is not None and not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x = x
                best_fx = fx
                best_abs = abs(fx)
            if success:
                return x

    # 6. Try secant from best point again
    if counter[0] < _CALL_BUDGET - 10 and _time_left() > 1.0:
        x, fx, success = _newton_secant(f, best_x, counter)
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x = x
                best_fx = fx
                best_abs = abs(fx)
            if success:
                return x

    if math.isnan(best_x):
        return x0

    return best_x
