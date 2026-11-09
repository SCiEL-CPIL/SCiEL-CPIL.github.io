"""Root finder for 3 fixed oscillatory equations with call-budget awareness."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, starting near x0.
    Uses a hybrid approach:
      1. Evaluate at x0; if good enough, return immediately.
      2. Try a small number of secant/Newton steps.
      3. If not converged, do a local bisection bracket search around x0.
      4. Track best (smallest |f|) throughout; return best if deadline approaches.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    tol = 1e-10  # tighter than required 1e-8

    x = float(x0)
    
    # Track best
    best_x = x
    try:
        best_f = f(x)
        best_abs = abs(best_f) if (best_f is not None and math.isfinite(best_f)) else float('inf')
    except Exception:
        best_f = float('inf')
        best_abs = float('inf')
    
    if best_abs <= tol:
        return best_x

    # Phase 1: Secant method (uses 2 points, no derivative estimation)
    # Start with x0 and a small perturbation
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    x2 = x - h
    
    try:
        f1 = f(x1)
        if f1 is not None and math.isfinite(f1) and abs(f1) < best_abs:
            best_abs = abs(f1)
            best_x = x1
            best_f = f1
        if best_abs <= tol:
            return best_x
    except Exception:
        f1 = float('inf')
    
    try:
        f2 = f(x2)
        if f2 is not None and math.isfinite(f2) and abs(f2) < best_abs:
            best_abs = abs(f2)
            best_x = x2
            best_f = f2
        if best_abs <= tol:
            return best_x
    except Exception:
        f2 = float('inf')
    
    # Try secant iterations
    a, fa = x1, f1
    b, fb = x2, f2
    
    if not (math.isfinite(fa) and math.isfinite(fb)):
        # One of them failed; use x0 and one valid point
        if math.isfinite(fa) and math.isfinite(best_f):
            a, fa = x1, f1
            b, fb = best_x, best_f
        elif math.isfinite(fb) and math.isfinite(best_f):
            a, fa = x2, f2
            b, fb = best_x, best_f
        else:
            # Fallback to bisection around x0
            return _bisection_search(f, x, h, best_x, best_abs, tol, deadline)
    
    for i in range(40):
        if time.time() > deadline - 5.0:
            break
        if fa == fb:
            break
        denom = fb - fa
        if denom == 0:
            break
        x_new = b - fb * (b - a) / denom
        if not math.isfinite(x_new):
            break
        try:
            f_new = f(x_new)
        except Exception:
            break
        if f_new is None or not math.isfinite(f_new):
            break
        if abs(f_new) < best_abs:
            best_abs = abs(f_new)
            best_x = x_new
            best_f = f_new
        if best_abs <= tol:
            return best_x
        a, fa = b, fb
        b, fb = x_new, f_new
    
    if best_abs <= tol:
        return best_x
    
    # Phase 2: Bisection search to find a sign change, then bisect
    return _bisection_search(f, x, h, best_x, best_abs, tol, deadline)


def _bisection_search(f, x0, h, best_x, best_abs, tol, deadline):
    """
    Search for a sign change near x0, then use bisection.
    Also try Newton if we can estimate derivative.
    """
    # Try to find a bracket [lo, hi] with sign change
    # Expand outward from x0
    scale = h
    for attempt in range(20):
        if time.time() > deadline - 3.0:
            break
        lo = x0 - scale
        hi = x0 + scale
        try:
            flo = f(lo)
            fhi = f(hi)
        except Exception:
            scale *= 2.0
            continue
        if flo is None or fhi is None or not math.isfinite(flo) or not math.isfinite(fhi):
            scale *= 2.0
            continue
        
        # Update best
        if abs(flo) < best_abs:
            best_abs = abs(flo)
            best_x = lo
        if abs(fhi) < best_abs:
            best_abs = abs(fhi)
            best_x = hi
        if best_abs <= tol:
            return best_x
        
        if flo * fhi < 0:
            # Found a bracket
            return _bisect(f, lo, hi, flo, fhi, best_x, best_abs, tol, deadline)
        
        scale *= 2.0
    
    # If no bracket found, try Newton from best_x
    return _newton_from(f, best_x, best_abs, tol, deadline)


def _bisect(f, lo, hi, flo, fhi, best_x, best_abs, tol, deadline):
    """Bisection within [lo, hi] where f(lo)*f(hi) < 0."""
    for i in range(80):
        if time.time() > deadline - 2.0:
            break
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        try:
            fmid = f(mid)
        except Exception:
            break
        if fmid is None or not math.isfinite(fmid):
            break
        if abs(fmid) < best_abs:
            best_abs = abs(fmid)
            best_x = mid
        if best_abs <= tol:
            return best_x
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    
    return best_x


def _newton_from(f, x, best_abs, tol, deadline):
    """Newton's method from x with finite-difference derivative."""
    x = float(x)
    for i in range(50):
        if time.time() > deadline - 2.0:
            break
        try:
            fx = f(x)
        except Exception:
            break
        if fx is None or not math.isfinite(fx):
            break
        if abs(fx) < best_abs:
            best_abs = abs(fx)
        if best_abs <= tol:
            return x
        
        h = 1e-6 * max(1.0, abs(x))
        try:
            fxp = f(x + h)
        except Exception:
            break
        if fxp is None or not math.isfinite(fxp):
            break
        d = (fxp - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        # Damping: if step is too large, take a smaller step
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            x_new = x + 0.5 * math.copysign(1.0, -fx / d) * min(abs(x_new - x), 1.0 * max(1.0, abs(x)))
        x = x_new
    
    return x
# EVOLVE-BLOCK-END
