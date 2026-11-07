"""Robust root finder for oscillatory equations with call-budget awareness."""
import math
import time
from scipy.optimize import brentq

# Safety margin: evaluator kills at 150s, use 140s
_DEADLINE = time.time() + 140.0
_CALL_BUDGET_PER_EQ = 480  # Stay under 500


def _time_left():
    return _DEADLINE - time.time()


def _wrap(f, counter, max_calls):
    """Wrap f to count calls and enforce budget."""
    def wrapped(x):
        counter[0] += 1
        if counter[0] > max_calls:
            raise RuntimeError("call budget exceeded")
        return f(x)
    return wrapped


def _safe_eval(f, x):
    """Evaluate f(x), returning (value, is_finite)."""
    try:
        val = f(x)
        if not math.isfinite(val):
            return val, False
        return val, True
    except Exception:
        return float('nan'), False


def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 for a given oscillatory function f."""
    counter = [0]
    max_calls = _CALL_BUDGET_PER_EQ
    wf = _wrap(f, counter, max_calls)
    
    best_x = float(x0)
    best_abs = float('inf')
    
    def _update_best(x, fx):
        nonlocal best_x, best_abs
        if math.isfinite(fx) and abs(fx) < best_abs:
            best_x = x
            best_abs = abs(fx)
        if best_abs <= 1e-8:
            return True
        return False
    
    # Initial evaluation
    fx0, fin0 = _safe_eval(wf, x0)
    if fin0:
        _update_best(x0, fx0)
        if best_abs <= 1e-8:
            return best_x
        if counter[0] >= max_calls:
            return best_x
    
    # Strategy 1: Try to bracket a root near x0 using expanding intervals
    # For oscillatory functions, roots are likely nearby
    
    # First, try a small neighborhood scan to find a sign change
    base_h = max(1e-6, abs(x0) * 1e-6)
    
    # Try multiple scales for bracketing
    scales = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    bracket_found = False
    a_bracket = None
    b_bracket = None
    fa = None
    fb = None
    
    for scale in scales:
        if counter[0] >= max_calls or _time_left() < 5.0:
            break
        h = scale * max(1.0, abs(x0))
        # Try points around x0
        candidates = []
        for i in range(1, 7):
            candidates.append(x0 + i * h)
            candidates.append(x0 - i * h)
        
        # Also try x0 itself as one endpoint
        points = [x0] + candidates
        fvals = []
        for p in points:
            if counter[0] >= max_calls:
                break
            val, fin = _safe_eval(wf, p)
            if fin:
                fvals.append((p, val))
                if abs(val) < best_abs:
                    best_x = p
                    best_abs = abs(val)
            else:
                fvals.append((p, None))
        
        # Check for sign change between consecutive points
        for i in range(len(fvals) - 1):
            p1, v1 = fvals[i]
            p2, v2 = fvals[i + 1]
            if v1 is None or v2 is None:
                continue
            if v1 == 0:
                return p1
            if v2 == 0:
                return p2
            if v1 * v2 < 0:
                a_bracket, b_bracket = p1, p2
                fa, fb = v1, v2
                bracket_found = True
                break
        if bracket_found:
            break
    
    # Strategy 2: If bracket found, use brentq
    if bracket_found:
        try:
            # Use a wrapper that respects our counter
            def brentq_f(x):
                return wf(x)
            
            result = brentq(brentq_f, a_bracket, b_bracket, 
                           xtol=1e-14, rtol=1e-14, maxiter=100)
            if math.isfinite(result):
                # Verify with our counted wrapper
                fr, fin = _safe_eval(wf, result)
                if fin and abs(fr) <= 1e-8:
                    return result
                # Even if not perfect, return it if it's the best
                if fin and abs(fr) < best_abs:
                    best_x = result
                    best_abs = abs(fr)
        except Exception:
            pass
    
    # Strategy 3: If no bracket or brentq failed, try Newton's method with fallback
    if not bracket_found or best_abs > 1e-8:
        x = float(x0)
        for _ in range(40):
            if counter[0] >= max_calls or _time_left() < 3.0:
                break
            fx, fin = _safe_eval(wf, x)
            if not fin:
                break
            if abs(fx) < best_abs:
                best_x = x
                best_abs = abs(fx)
            if best_abs <= 1e-8:
                return best_x
            
            h = 1e-7 * max(1.0, abs(x))
            fxh, finh = _safe_eval(wf, x + h)
            if not finh:
                break
            d = (fxh - fx) / h
            if d == 0 or not math.isfinite(d):
                # Try central difference
                fxm, finm = _safe_eval(wf, x - h)
                if not finm:
                    break
                d = (fxh - fxm) / (2 * h)
                if d == 0 or not math.isfinite(d):
                    break
            
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            
            # Damped Newton step if it goes too far
            step = x_new - x
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                x_new = x + max_step * math.copysign(1.0, step)
            
            x = x_new
    
    # Strategy 4: If still not converged, do a finer local scan around best_x
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 2.0:
        # Scan around best_x with decreasing step sizes
        for scale in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            if counter[0] >= max_calls or _time_left() < 2.0:
                break
            h = scale * max(1.0, abs(best_x))
            for i in range(-5, 6):
                if counter[0] >= max_calls:
                    break
                p = best_x + i * h
                val, fin = _safe_eval(wf, p)
                if fin and abs(val) < best_abs:
                    best_x = p
                    best_abs = abs(val)
                if best_abs <= 1e-8:
                    return best_x
    
    # Final: return the best we found
    return best_x
