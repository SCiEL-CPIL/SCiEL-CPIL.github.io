"""Root finder for smooth polynomial equations with call-budget awareness."""
import math
import time
import numpy as np

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _find_polynomial_roots(coeffs):
    """Find real roots of polynomial given by coefficients (highest degree first)."""
    if len(coeffs) == 0:
        return []
    if len(coeffs) == 1:
        return []
    if len(coeffs) == 2:
        a, b = coeffs
        if abs(a) < 1e-15:
            return []
        return [-b / a]
    try:
        roots = np.roots(coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        return real_roots
    except Exception:
        return []


def _safeguarded_secant(f, x0, x1, f0, f1, lo, hi, f_lo, f_hi, max_iter=50):
    """Safeguarded secant method with bisection fallback within [lo, hi]."""
    best_x = x0 if abs(f0) < abs(f1) else x1
    best_f = min(abs(f0), abs(f1))
    
    x_prev, x_curr = x0, x1
    f_prev, f_curr = f0, f1
    
    for _ in range(max_iter):
        if _time_left() < 3.0:
            break
        
        # Ensure |f_curr| <= |f_prev|
        if abs(f_curr) > abs(f_prev):
            x_prev, x_curr = x_curr, x_prev
            f_prev, f_curr = f_curr, f_prev
        
        if abs(f_curr) <= 1e-8:
            return x_curr
        
        # Secant step
        if abs(f_curr - f_prev) < 1e-30:
            x_new = (lo + hi) / 2.0
        else:
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # If outside bracket, use bisection
        if not (lo < x_new < hi):
            x_new = (lo + hi) / 2.0
        
        f_new = f(x_new)
        if not _is_finite(f_new):
            break
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        if abs(f_new) < best_f:
            best_x, best_f = x_new, abs(f_new)
        
        # Update bracket
        if f_new * f_lo < 0:
            hi, f_hi = x_new, f_new
        else:
            lo, f_lo = x_new, f_new
        
        x_prev, x_curr = x_curr, x_new
        f_prev, f_curr = f_curr, f_new
        
        if abs(hi - lo) < 1e-14 * max(1.0, abs(x_new)):
            break
    
    # Return best candidate
    candidates = [x_curr, (lo + hi) / 2.0]
    for cand in candidates:
        if _is_finite(cand):
            fc = f(cand)
            if _is_finite(fc) and abs(fc) < best_f:
                best_x, best_f = cand, abs(fc)
    
    return best_x if best_f <= 1e-8 else None


def _newton_polish(f, x, f_val, max_iter=20):
    """Newton's method polish for high-precision convergence."""
    x_curr = x
    f_curr = f_val
    
    for _ in range(max_iter):
        if _time_left() < 2.0:
            break
        if abs(f_curr) <= 1e-10:
            break
        
        h = 1e-7 * max(1.0, abs(x_curr))
        f_plus = f(x_curr + h)
        if not _is_finite(f_plus):
            break
        
        df = (f_plus - f_curr) / h
        if abs(df) < 1e-15 or not _is_finite(df):
            break
        
        x_new = x_curr - f_curr / df
        if not _is_finite(x_new):
            break
        
        # Damping if step too large
        if abs(x_new - x_curr) > 5 * max(1.0, abs(x_curr)):
            x_new = x_curr + 0.5 * (x_new - x_curr)
        
        f_new = f(x_new)
        if not _is_finite(f_new):
            break
        
        if abs(f_new) < abs(f_curr):
            x_curr, f_curr = x_new, f_new
        else:
            break
    
    return x_curr, f_curr


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Strategy: 
    1. Evaluate at x0 and nearby points
    2. Fit low-degree polynomial to predict roots
    3. Refine best candidates with safeguarded secant
    4. Polish with Newton
    """
    if not _is_finite(x0):
        x0 = 0.0
    x = float(x0)
    
    best_x = x
    best_f = f(x)
    if not _is_finite(best_f):
        return 0.0
    if abs(best_f) <= 1e-8:
        return best_x
    
    # Phase 1: Sample points for polynomial interpolation
    # Use geometrically spaced points around x0
    interp_points = [(x, best_f)]
    offsets = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]
    for off in offsets:
        if _time_left() < 8.0:
            break
        x_p = x + off
        f_p = f(x_p)
        if not _is_finite(f_p):
            continue
        interp_points.append((x_p, f_p))
        if abs(f_p) <= 1e-8:
            return x_p
        if len(interp_points) >= 5:
            break
    
    # Phase 2: Fit polynomial and find its roots
    candidates = []
    if len(interp_points) >= 3:
        xs = [p[0] for p in interp_points]
        ys = [p[1] for p in interp_points]
        n = len(xs)
        deg = min(n - 1, 4)
        
        try:
            coeffs = np.polyfit(xs, ys, deg)
            real_roots = _find_polynomial_roots(coeffs)
            
            for root in real_roots:
                if _time_left() < 5.0:
                    break
                if not _is_finite(root):
                    continue
                if abs(root - x) > 1e8:
                    continue
                candidates.append(root)
        except Exception:
            pass
    
    # Also add bracket endpoints as candidates if we can find them
    # Quick bracket search
    bracket_found = False
    step = 1.0
    for direction in [1, -1]:
        if bracket_found:
            break
        for _ in range(15):
            if _time_left() < 6.0:
                break
            x_try = x + direction * step
            f_try = f(x_try)
            if not _is_finite(f_try):
                step *= 2.0
                continue
            if abs(f_try) <= 1e-8:
                return x_try
            if best_f * f_try < 0:
                a, b = (x, x_try) if direction == 1 else (x_try, x)
                fa, fb = (best_f, f_try) if direction == 1 else (f_try, best_f)
                candidates.append((a + b) / 2.0)
                # Refine with safeguarded secant
                result = _safeguarded_secant(f, a, b, fa, fb, a, b, fa, fb)
                if result is not None:
                    return result
                bracket_found = True
                break
            step *= 2.0
            if step > 1e10:
                break
    
    # Phase 3: Evaluate all candidates and refine the best
    best_candidate = None
    best_candidate_f = float('inf')
    
    # Add x0 neighborhood points as additional candidates
    extra_candidates = [x + 0.1, x - 0.1, x + 0.01, x - 0.01]
    all_candidates = candidates + extra_candidates
    
    for cand in all_candidates:
        if _time_left() < 4.0:
            break
        if not _is_finite(cand):
            continue
        fc = f(cand)
        if not _is_finite(fc):
            continue
        if abs(fc) < best_candidate_f:
            best_candidate = cand
            best_candidate_f = abs(fc)
            if abs(fc) <= 1e-8:
                return cand
    
    if best_candidate is not None and best_candidate_f < abs(best_f):
        best_x, best_f = best_candidate, best_candidate_f
    
    if abs(best_f) <= 1e-8:
        return best_x
    
    # Phase 4: Try to find a bracket around best_x and refine
    if _time_left() > 5.0:
        # Search for bracket around best_x
        h = max(0.1, abs(best_x) * 0.1)
        bracket_found = False
        for scale in [1, 2, 4, 8, 16]:
            if _time_left() < 3.0:
                break
            lo = best_x - h * scale
            hi = best_x + h * scale
            f_lo = f(lo)
            f_hi = f(hi)
            if not _is_finite(f_lo) or not _is_finite(f_hi):
                continue
            if f_lo * f_hi < 0:
                result = _safeguarded_secant(f, lo, hi, f_lo, f_hi, lo, hi, f_lo, f_hi)
                if result is not None:
                    return result
                bracket_found = True
                break
            if abs(f_lo) < abs(best_f):
                best_x, best_f = lo, f_lo
            if abs(f_hi) < abs(best_f):
                best_x, best_f = hi, f_hi
    
    # Phase 5: Newton polish
    if _time_left() > 3.0 and abs(best_f) > 1e-10:
        x_polished, f_polished = _newton_polish(f, best_x, best_f)
        if _is_finite(f_polished) and abs(f_polished) < abs(best_f):
            best_x, best_f = x_polished, f_polished
    
    if abs(best_f) <= 1e-8:
        return best_x
    
    return best_x
