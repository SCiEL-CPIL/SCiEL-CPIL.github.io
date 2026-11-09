"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_T0 = time.time()
_LIMIT = 110.0
_CALL_LIMIT = 495  # stay well under 500

def _left():
    return _LIMIT - (time.time() - _T0)

def _fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def _poly_interp_root(xs, fs, degree=None):
    """Fit a polynomial to (xs, fs) and find its real roots closest to the data."""
    n = len(xs)
    if n < 2:
        return None
    if degree is None:
        degree = min(n - 1, 4)
    # Use numpy polyfit
    try:
        coeffs = np.polyfit(xs, fs, degree)
        roots = np.roots(coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        if not real_roots:
            return None
        # Return the real root closest to the mean of xs
        cx = sum(xs) / len(xs)
        real_roots.sort(key=lambda r: abs(r - cx))
        return real_roots[0]
    except Exception:
        return None

def _secant_step(x1, f1, x2, f2):
    """Compute secant step from two points."""
    if f1 == f2:
        return None
    return x2 - f2 * (x2 - x1) / (f2 - f1)

def find_root(f, x0):
    x0 = float(x0)
    best_x = x0
    best_fx = float('inf')
    call_count = 0
    
    def _eval(x):
        nonlocal call_count, best_x, best_fx
        call_count += 1
        try:
            fx = f(x)
        except Exception:
            return None
        if not _fin(fx):
            return None
        if abs(fx) < abs(best_fx):
            best_fx, best_x = fx, x
        return fx
    
    # Evaluate starting point
    fx0 = _eval(x0)
    if fx0 is not None and abs(fx0) <= 1e-8:
        return x0
    
    # Strategy 1: Geometric spread sampling to find sign change quickly
    if call_count < _CALL_LIMIT and _left() > 2.0:
        base_scale = max(1.0, abs(x0))
        # Geometric spread: sample at exponentially increasing distances
        samples = [(x0, fx0)]
        
        # Try both sides with geometric progression
        for scale in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
            if call_count >= _CALL_LIMIT or _left() < 2.0:
                break
            for sign in [1, -1]:
                if call_count >= _CALL_LIMIT or _left() < 2.0:
                    break
                x_try = x0 + sign * scale * base_scale
                if not _fin(x_try):
                    continue
                fx_try = _eval(x_try)
                if fx_try is not None:
                    if abs(fx_try) <= 1e-8:
                        return x_try
                    samples.append((x_try, fx_try))
        
        # Look for sign changes in samples
        samples.sort(key=lambda p: p[0])
        bracket = None
        for i in range(len(samples) - 1):
            if samples[i][1] * samples[i+1][1] < 0:
                bracket = (samples[i], samples[i+1])
                break
        
        if bracket is not None:
            # Use bisection + secant hybrid to converge
            a, fa = bracket[0]
            b, fb = bracket[1]
            
            # Secant iterations within bracket
            for i in range(30):
                if call_count >= _CALL_LIMIT or _left() < 1.0:
                    break
                if abs(fa) <= 1e-8:
                    return a
                if abs(fb) <= 1e-8:
                    return b
                
                # Secant step
                x_sec = _secant_step(a, fa, b, fb)
                if x_sec is None or not _fin(x_sec):
                    # Fall back to bisection
                    x_sec = (a + b) / 2.0
                else:
                    # Ensure secant point is within bracket
                    if not (min(a, b) < x_sec < max(a, b)):
                        x_sec = (a + b) / 2.0
                
                fx_sec = _eval(x_sec)
                if fx_sec is None:
                    break
                if abs(fx_sec) <= 1e-8:
                    return x_sec
                
                if fa * fx_sec < 0:
                    b, fb = x_sec, fx_sec
                else:
                    a, fa = x_sec, fx_sec
                
                if abs(b - a) < 1e-14:
                    break
            
            # Final polish with high precision
            if call_count < _CALL_LIMIT and _left() > 0.5:
                x_mid = (a + b) / 2.0
                fx_mid = _eval(x_mid)
                if fx_mid is not None and abs(fx_mid) <= 1e-8:
                    return x_mid
                # Use the point with smaller |f|
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b
    
    # Strategy 2: Polynomial interpolation if no bracket found
    if call_count < _CALL_LIMIT and _left() > 2.0 and abs(best_fx) > 1e-8:
        # Collect points from previous sampling
        points = []
        # Re-evaluate key points if needed
        base_scale = max(1.0, abs(x0))
        for off in [0, 0.1, -0.1, 0.3, -0.3, 1.0, -1.0]:
            if call_count >= _CALL_LIMIT or _left() < 2.0:
                break
            x_try = x0 + off * base_scale
            if not _fin(x_try):
                continue
            fx_try = _eval(x_try)
            if fx_try is not None:
                if abs(fx_try) <= 1e-8:
                    return x_try
                points.append((x_try, fx_try))
        
        if len(points) >= 3:
            # Try polynomial interpolation with degree 2 and 3
            for deg in [2, 3]:
                if call_count >= _CALL_LIMIT or _left() < 2.0:
                    break
                if len(points) < deg + 1:
                    continue
                
                # Select points: prioritize those with sign changes or small |f|
                # Sort by x to maintain order for interpolation
                pts_sorted = sorted(points, key=lambda p: p[0])
                
                # Try different subsets
                subsets = []
                # Consecutive points
                for i in range(len(pts_sorted) - deg):
                    subsets.append(pts_sorted[i:i+deg+1])
                # Points closest to zero
                pts_by_f = sorted(points, key=lambda p: abs(p[1]))
                subsets.append(pts_by_f[:deg+1])
                
                tried_roots = set()
                for subset in subsets:
                    if call_count >= _CALL_LIMIT or _left() < 2.0:
                        break
                    xs = [p[0] for p in subset]
                    fs = [p[1] for p in subset]
                    
                    root_est = _poly_interp_root(xs, fs, degree=deg)
                    if root_est is not None and _fin(root_est):
                        if root_est in tried_roots:
                            continue
                        tried_roots.add(root_est)
                        
                        fx_root = _eval(root_est)
                        if fx_root is not None:
                            if abs(fx_root) <= 1e-8:
                                return root_est
                            # If reasonably close, refine with secant
                            if abs(fx_root) < 1e-4:
                                # Use secant method from two nearby points
                                # Find another point close to root_est
                                other_pts = [(p[0], p[1]) for p in points if abs(p[0] - root_est) > 1e-10]
                                if other_pts:
                                    # Pick the closest other point
                                    other_pts.sort(key=lambda p: abs(p[0] - root_est))
                                    x1, f1 = other_pts[0]
                                    x2, f2 = root_est, fx_root
                                    
                                    for i in range(20):
                                        if call_count >= _CALL_LIMIT or _left() < 1.0:
                                            break
                                        x_new = _secant_step(x1, f1, x2, f2)
                                        if x_new is None or not _fin(x_new):
                                            break
                                        fx_new = _eval(x_new)
                                        if fx_new is None:
                                            break
                                        if abs(fx_new) <= 1e-8:
                                            return x_new
                                        x1, f1 = x2, f2
                                        x2, f2 = x_new, fx_new
                                break
    
    # Strategy 3: Secant method from best point
    if call_count < _CALL_LIMIT and _left() > 2.0 and abs(best_fx) > 1e-8:
        x1 = best_x
        f1 = best_fx
        # Get a second point
        offset = 0.1 * max(1.0, abs(x1))
        x2 = x1 + offset
        f2 = _eval(x2)
        
        if f2 is None:
            x2 = x1 - offset
            f2 = _eval(x2)
        
        if f2 is not None and f2 != f1:
            for i in range(30):
                if call_count >= _CALL_LIMIT or _left() < 2.0:
                    break
                if abs(f1) <= 1e-8:
                    return x1
                if abs(f2) <= 1e-8:
                    return x2
                
                x_new = _secant_step(x1, f1, x2, f2)
                if x_new is None or not _fin(x_new):
                    break
                f_new = _eval(x_new)
                if f_new is None:
                    break
                if abs(f_new) <= 1e-8:
                    return x_new
                
                # Keep the two points with smallest |f|
                if abs(f_new) < abs(f1):
                    x1, f1 = x2, f2
                    x2, f2 = x_new, f_new
                else:
                    x1, f1 = x2, f2
                    x2, f2 = x_new, f_new
    
    # Strategy 4: Bisection with wider bracket
    if call_count < _CALL_LIMIT and _left() > 2.0 and abs(best_fx) > 1e-8:
        for scale in [0.5, 2.0, 5.0, 20.0, 50.0, 200.0, 500.0]:
            if call_count >= _CALL_LIMIT or _left() < 2.0:
                break
            a = x0 - scale
            b = x0 + scale
            fa = _eval(a)
            fb = _eval(b)
            
            if fa is not None and abs(fa) <= 1e-8:
                return a
            if fb is not None and abs(fb) <= 1e-8:
                return b
            
            if fa is not None and fb is not None and fa * fb < 0:
                for i in range(50):
                    if call_count >= _CALL_LIMIT or _left() < 1.0:
                        break
                    mid = (a + b) / 2.0
                    fm = _eval(mid)
                    if fm is None:
                        break
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                    if abs(b - a) < 1e-14:
                        break
                break
    
    # Strategy 5: Local minimization fallback
    if call_count < _CALL_LIMIT and _left() > 2.0 and abs(best_fx) > 1e-8:
        x = best_x
        step = 0.1 * max(1.0, abs(x))
        for i in range(15):
            if call_count >= _CALL_LIMIT or _left() < 1.0:
                break
            best_local = (x, abs(best_fx))
            for dx in [-step, step, -0.5*step, 0.5*step, -2*step, 2*step]:
                if call_count >= _CALL_LIMIT:
                    break
                x_try = x + dx
                if not _fin(x_try):
                    continue
                fx_try = _eval(x_try)
                if fx_try is None:
                    continue
                if abs(fx_try) <= 1e-8:
                    return x_try
                if abs(fx_try) < best_local[1]:
                    best_local = (x_try, abs(fx_try))
            if best_local[0] == x:
                step *= 0.5
                if step < 1e-10:
                    break
            else:
                x = best_local[0]
    
    return best_x
# EVOLVE-BLOCK-END
