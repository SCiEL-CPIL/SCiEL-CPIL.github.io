import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 450

def _time_left():
    return _DEADLINE - time.time()

def _safe_call(f, x):
    try:
        val = f(x)
        if not math.isfinite(val):
            return float('nan')
        return val
    except Exception:
        return float('nan')

def _secant_step(f, x0, fx0, counter, max_iter=5):
    """
    Perform a few secant steps.
    Returns (x, fx, success, calls_made)
    """
    x1 = x0 + 1e-4 * max(1.0, abs(x0))
    fx1 = _safe_call(f, x1)
    counter[0] += 1
    if math.isnan(fx1):
        return x0, fx0, False, 1
    if abs(fx1) <= 1e-8:
        return x1, fx1, True, 1
    
    best_x = x0 if abs(fx0) < abs(fx1) else x1
    best_fx = fx0 if abs(fx0) < abs(fx1) else fx1
    best_abs = min(abs(fx0), abs(fx1))
    
    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fx0 == fx1:
            break
        step = (x1 - x0) * fx0 / (fx1 - fx0)
        # Limit step size to prevent divergence
        max_step = 1.0 + abs(x0)
        if abs(step) > max_step:
            step = max_step * (1 if step > 0 else -1)
        x_new = x0 - step
        if not math.isfinite(x_new):
            break
        
        x0, fx0 = x1, fx1
        x1 = x_new
        fx1 = _safe_call(f, x1)
        counter[0] += 1
        if math.isnan(fx1):
            break
        if abs(fx1) <= 1e-8:
            return x1, fx1, True, counter[0]
        if abs(fx0) <= 1e-8:
            return x0, fx0, True, counter[0]
        
        if abs(fx1) < best_abs:
            best_x = x1
            best_fx = fx1
            best_abs = abs(fx1)
        if abs(fx0) < best_abs:
            best_x = x0
            best_fx = fx0
            best_abs = abs(fx0)
            
    return best_x, best_fx, best_abs <= 1e-8, counter[0]

def _find_bracket(f, x0, fx0, counter):
    """
    Find a bracket [a, b] with f(a)*f(b) < 0.
    Strategy:
    1. Small local steps.
    2. Geometric expansion.
    """
    if math.isnan(fx0):
        return None
    if abs(fx0) <= 1e-8:
        return x0, x0, fx0, fx0

    a = x0
    fa = fx0
    
    # 1. Try small local steps to catch nearby oscillations
    base_step = 1e-3 * max(1.0, abs(x0))
    for i in range(1, 6):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        step = base_step * i
        # Try right
        b = x0 + step
        fb = _safe_call(f, b)
        counter[0] += 1
        if not math.isnan(fb):
            if abs(fb) <= 1e-8:
                return b, b, fb, fb
            if fa * fb < 0:
                return a, b, fa, fb
        # Try left
        b = x0 - step
        fb = _safe_call(f, b)
        counter[0] += 1
        if not math.isnan(fb):
            if abs(fb) <= 1e-8:
                return b, b, fb, fb
            if fa * fb < 0:
                return b, a, fb, fa
        
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break

    # 2. Geometric expansion
    step = base_step * 10
    for i in range(20):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        # Alternate sides to cover more ground efficiently
        if i % 2 == 0:
            b = x0 + step
            fb = _safe_call(f, b)
            counter[0] += 1
            if not math.isnan(fb):
                if abs(fb) <= 1e-8:
                    return b, b, fb, fb
                if fa * fb < 0:
                    return a, b, fa, fb
        else:
            b = x0 - step
            fb = _safe_call(f, b)
            counter[0] += 1
            if not math.isnan(fb):
                if abs(fb) <= 1e-8:
                    return b, b, fb, fb
                if fa * fb < 0:
                    return b, a, fb, fa
        step *= 2.0
        if step > 1e6:
            break

    return None

def _brentq_solve(f, a, b, fa, fb, counter):
    """
    Use scipy brentq if available, else fallback to bisection/secant hybrid.
    """
    if _HAS_SCIPY:
        try:
            # xtol=1e-12, rtol=1e-12 is very precise
            x_br = _scipy_brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            fx_br = _safe_call(f, x_br)
            counter[0] += 1
            if not math.isnan(fx_br):
                return x_br, fx_br, abs(fx_br) <= 1e-8
        except Exception:
            pass
    
    # Fallback: Simple Bisection
    # We know fa*fb < 0
    for _ in range(50):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        c = (a + b) / 2.0
        fc = _safe_call(f, c)
        counter[0] += 1
        if math.isnan(fc):
            break
        if abs(fc) <= 1e-8:
            return c, fc, True
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        if abs(b - a) < 1e-12 * max(1.0, abs(c)):
            break
    c = (a + b) / 2.0
    fc = _safe_call(f, c)
    counter[0] += 1
    return c, fc, abs(fc) <= 1e-8

def find_root(f, x0):
    counter = [0]
    x0 = float(x0)
    
    # 1. Initial evaluation
    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if not math.isnan(fx0) and abs(fx0) <= 1e-8:
        return x0
    
    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0) if not math.isnan(fx0) else float('inf')
    
    # 2. Try Secant for a few steps (fast if it works)
    x_sec, fx_sec, success, _ = _secant_step(f, x0, fx0, counter, max_iter=5)
    if not math.isnan(fx_sec):
        if abs(fx_sec) < best_abs:
            best_x = x_sec
            best_fx = fx_sec
            best_abs = abs(fx_sec)
        if success:
            return best_x
            
    # 3. Find a bracket from the best point found so far
    bracket = _find_bracket(f, best_x, best_fx, counter)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a == b and abs(fa) <= 1e-8:
            return a
        if fa * fb < 0:
            x_br, fx_br, success = _brentq_solve(f, a, b, fa, fb, counter)
            if not math.isnan(fx_br):
                if abs(fx_br) < best_abs:
                    best_x = x_br
                    best_fx = fx_br
                    best_abs = abs(fx_br)
                if success:
                    return best_x

    # 4. If no bracket found, try a wider scan from x0
    if best_abs > 1e-8:
        # Scan a wider range
        scan_points = [x0 + 0.1, x0 - 0.1, x0 + 1.0, x0 - 1.0, x0 + 10.0, x0 - 10.0]
        for pt in scan_points:
            if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                break
            f_pt = _safe_call(f, pt)
            counter[0] += 1
            if math.isnan(f_pt):
                continue
            if abs(f_pt) < best_abs:
                best_x = pt
                best_fx = f_pt
                best_abs = abs(f_pt)
            if abs(f_pt) <= 1e-8:
                return pt
            
            # Try to bracket from this new point
            bracket = _find_bracket(f, pt, f_pt, counter)
            if bracket is not None:
                a, b, fa, fb = bracket
                if a == b and abs(fa) <= 1e-8:
                    return a
                if fa * fb < 0:
                    x_br, fx_br, success = _brentq_solve(f, a, b, fa, fb, counter)
                    if not math.isnan(fx_br):
                        if abs(fx_br) < best_abs:
                            best_x = x_br
                            best_fx = fx_br
                            best_abs = abs(fx_br)
                        if success:
                            return best_x

    return best_x
