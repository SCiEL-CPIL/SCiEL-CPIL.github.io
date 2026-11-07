import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 495

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

    if math.isnan(fx0):
        # If f(x0) is NaN, try nearby points
        for h in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2]:
            x_try = x0 + h
            fx_try = _safe_call(f, x_try)
            counter[0] += 1
            if not math.isnan(fx_try):
                if abs(fx_try) < best_abs:
                    best_x = x_try
                    best_fx = fx_try
                    best_abs = abs(fx_try)
                if abs(fx_try) <= 1e-8:
                    return x_try
        # If still NaN, we can't do much, return best guess or x0
        return best_x if not math.isnan(best_x) else x0

    # 2. Try Newton step to get a better point or bracket
    # Estimate derivative
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = _safe_call(f, x1)
    counter[0] += 1
    
    if not math.isnan(fx1):
        if abs(fx1) < best_abs:
            best_x = x1
            best_fx = fx1
            best_abs = abs(fx1)
        if abs(fx1) <= 1e-8:
            return x1
        
        # Newton step
        if fx1 != fx0:
            slope = (fx1 - fx0) / (x1 - x0)
            if slope != 0:
                x_new = x0 - fx0 / slope
                if math.isfinite(x_new):
                    fx_new = _safe_call(f, x_new)
                    counter[0] += 1
                    if not math.isnan(fx_new):
                        if abs(fx_new) < best_abs:
                            best_x = x_new
                            best_fx = fx_new
                            best_abs = abs(fx_new)
                        if abs(fx_new) <= 1e-8:
                            return x_new
                        
                        # Check if we bracketed a root
                        # Points: x0, x1, x_new
                        # Check pairs (x0, x_new), (x0, x1), (x1, x_new)
                        points = [(x0, fx0), (x1, fx1), (x_new, fx_new)]
                        for i in range(len(points)):
                            for j in range(i+1, len(points)):
                                xa, fa = points[i]
                                xb, fb = points[j]
                                if fa * fb < 0:
                                    # Found bracket
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
                                            x_root = _scipy_brentq(wrapped_f, xa, xb, xtol=1e-14, rtol=1e-14, maxiter=50)
                                            fx_root = _safe_call(f, x_root)
                                            counter[0] += 1
                                            if not math.isnan(fx_root) and abs(fx_root) <= 1e-8:
                                                return x_root
                                        except Exception:
                                            pass
                                    # Fallback if scipy fails or not available
                                    # Simple bisection
                                    a, b = xa, xb
                                    fa, fb = fx0 if xa==x0 else (fx1 if xa==x1 else fx_new), fx0 if xb==x0 else (fx1 if xb==x1 else fx_new)
                                    # Note: fa, fb assignment above is messy, let's just use the values
                                    fa = f(xa) if False else (fx0 if xa==x0 else (fx1 if xa==x1 else fx_new))
                                    fb = f(xb) if False else (fx0 if xb==x0 else (fx1 if xb==x1 else fx_new))
                                    for _ in range(50):
                                        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                                            break
                                        c = (a + b) / 2.0
                                        fc = _safe_call(f, c)
                                        counter[0] += 1
                                        if math.isnan(fc):
                                            break
                                        if abs(fc) <= 1e-8:
                                            return c
                                        if fa * fc < 0:
                                            b = c
                                            fb = fc
                                        else:
                                            a = c
                                            fa = fc
                                    return (a+b)/2.0

    # 3. If Newton didn't find a bracket, do standard exponential bracketing
    # Start from x0
    a = x0
    fa = fx0
    step = 1e-3
    # Try expanding in + direction
    for _ in range(50):
        if counter[0] >= _CALL_BUDGET - 10 or _time_left() < 1.0:
            break
        b = a + step
        fb = _safe_call(f, b)
        counter[0] += 1
        if math.isnan(fb):
            break
        if fb * fa < 0:
            # Bracket found
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
                    x_root = _scipy_brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50)
                    fx_root = _safe_call(f, x_root)
                    counter[0] += 1
                    if not math.isnan(fx_root) and abs(fx_root) <= 1e-8:
                        return x_root
                except Exception:
                    pass
            # Fallback bisection
            for _ in range(50):
                if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                    break
                c = (a + b) / 2.0
                fc = _safe_call(f, c)
                counter[0] += 1
                if math.isnan(fc):
                    break
                if abs(fc) <= 1e-8:
                    return c
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            return (a+b)/2.0
        a = b
        fa = fb
        step *= 2.0
        if step > 1e6:
            break

    # Try expanding in - direction
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
        if fb * fa < 0:
            # Bracket found
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
                    x_root = _scipy_brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50)
                    fx_root = _safe_call(f, x_root)
                    counter[0] += 1
                    if not math.isnan(fx_root) and abs(fx_root) <= 1e-8:
                        return x_root
                except Exception:
                    pass
            # Fallback bisection
            for _ in range(50):
                if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                    break
                c = (a + b) / 2.0
                fc = _safe_call(f, c)
                counter[0] += 1
                if math.isnan(fc):
                    break
                if abs(fc) <= 1e-8:
                    return c
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            return (a+b)/2.0
        a = b
        fa = fb
        step *= 2.0
        if step > 1e6:
            break

    # If no bracket found, return best point
    return best_x
