"""Root finding for ill-conditioned equations: secant + inverse interpolation + manual Brent."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def find_root(f, x0):
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin before 150s limit
    max_evals = 500
    eval_count = [0]
    
    def safe_eval(x):
        if eval_count[0] >= max_evals:
            raise RuntimeError("Eval budget exceeded")
        if time.time() > deadline:
            raise RuntimeError("Time deadline exceeded")
        eval_count[0] += 1
        return f(x)
    
    def check_time():
        if time.time() > deadline:
            raise RuntimeError("Time deadline exceeded")
    
    x = float(x0)
    best_x = x
    best_f = safe_eval(x)
    best_abs = abs(best_f)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Keep history of (x, f(x)) for inverse interpolation
    history = [(x, best_f)]
    
    # Phase 1: Secant / Inverse interpolation with safeguarding
    a = None
    b = None
    fa = None
    fb = None
    
    for i in range(200):
        check_time()
        if eval_count[0] >= max_evals - 10:
            break
        
        if best_abs <= 1e-8:
            return best_x
        
        # Try inverse quadratic interpolation if we have 3 points
        if len(history) >= 3:
            x1, f1 = history[-1]
            x2, f2 = history[-2]
            x3, f3 = history[-3]
            
            # Check points are distinct
            if abs(x1 - x2) > 1e-15 and abs(x2 - x3) > 1e-15 and abs(x1 - x3) > 1e-15:
                # Inverse quadratic interpolation: solve for x where f(x)=0
                # Using Lagrange form for inverse: x = x1*L1(0) + x2*L2(0) + x3*L3(0)
                # where Li(t) = prod_{j!=i} (t - fj)/(fi - fj)
                try:
                    denom1 = (f1 - f2) * (f1 - f3)
                    denom2 = (f2 - f1) * (f2 - f3)
                    denom3 = (f3 - f1) * (f3 - f2)
                    
                    if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                        # L1(0) = (0-f2)*(0-f3) / ((f1-f2)*(f1-f3))
                        L1 = (f2 * f3) / denom1
                        L2 = (f1 * f3) / denom2
                        L3 = (f1 * f2) / denom3
                        
                        x_new = x1 * L1 + x2 * L2 + x3 * L3
                        
                        if np.isfinite(x_new):
                            # Safeguard: if x_new is outside [min,max] of recent points, use secant
                            x_min = min(x1, x2, x3)
                            x_max = max(x1, x2, x3)
                            # Allow some extrapolation but not too much
                            margin = 2.0 * (x_max - x_min) if x_max > x_min else 1.0
                            if x_new < x_min - margin or x_new > x_max + margin:
                                # Fall back to secant with two closest points
                                d_sec = (f1 - f2) / (x1 - x2)
                                if abs(d_sec) > 1e-30:
                                    x_new = x1 - f1 / d_sec
                                else:
                                    x_new = None
                            else:
                                x_new = x_new
                        else:
                            x_new = None
                    else:
                        x_new = None
                except (ZeroDivisionError, OverflowError):
                    x_new = None
                
                if x_new is None or not np.isfinite(x_new):
                    # Fall back to secant
                    if len(history) >= 2:
                        x1, f1 = history[-1]
                        x2, f2 = history[-2]
                        if abs(x1 - x2) > 1e-15:
                            d_sec = (f1 - f2) / (x1 - x2)
                            if abs(d_sec) > 1e-30:
                                x_new = x1 - f1 / d_sec
                            else:
                                x_new = None
                        else:
                            x_new = None
                    else:
                        x_new = None
            else:
                x_new = None
        elif len(history) >= 2:
            # Secant method
            x1, f1 = history[-1]
            x2, f2 = history[-2]
            if abs(x1 - x2) > 1e-15:
                d_sec = (f1 - f2) / (x1 - x2)
                if abs(d_sec) > 1e-30:
                    x_new = x1 - f1 / d_sec
                else:
                    x_new = None
            else:
                x_new = None
        else:
            # Only one point; use finite difference Newton
            h = 1e-6 * max(1.0, abs(x))
            try:
                fp = safe_eval(x + h)
                d = (fp - best_f) / h
                if abs(d) > 1e-30:
                    x_new = x - best_f / d
                else:
                    x_new = None
            except RuntimeError:
                break
        
        if x_new is None or not np.isfinite(x_new):
            break
        
        # Clip step to avoid runaway
        ref_x = history[-1][0]
        if abs(x_new - ref_x) > 20 * max(1.0, abs(ref_x)):
            x_new = ref_x + math.copysign(20 * max(1.0, abs(ref_x)), x_new - ref_x)
        
        fx_new = safe_eval(x_new)
        abs_new = abs(fx_new)
        
        # Update bracket
        if a is None:
            a = history[-1][0]
            fa = history[-1][1]
        
        if fa is not None and (fa > 0) != (fx_new > 0):
            # We have a sign change
            if b is None or (fb is not None and (fb > 0) != (fx_new > 0) is False):
                # Keep the bracket with smaller interval
                if b is not None:
                    if abs(a - b) > abs(a - x_new):
                        b = x_new
                        fb = fx_new
                    elif abs(a - b) > abs(b - x_new):
                        a = x_new
                        fa = fx_new
                else:
                    b = x_new
                    fb = fx_new
        elif b is not None and (fb > 0) != (fx_new > 0):
            b = x_new
            fb = fx_new
        
        # Update best
        if abs_new < best_abs:
            best_x = x_new
            best_f = fx_new
            best_abs = abs_new
        
        # Add to history (keep last 5 points)
        history.append((x_new, fx_new))
        if len(history) > 5:
            history.pop(0)
        
        # If we have a valid bracket and are making slow progress, use bisection
        if a is not None and b is not None and fa is not None and fb is not None:
            if (fa > 0) != (fb > 0):
                bracket_width = abs(a - b)
                if bracket_width < 1e-12 * max(1.0, abs(a), abs(b)):
                    # Bracket is very small, should be close to root
                    mid = (a + b) / 2.0
                    fm = safe_eval(mid)
                    if abs(fm) < best_abs:
                        best_x = mid
                        best_f = fm
                        best_abs = abs(fm)
                    history.append((mid, fm))
                    if len(history) > 5:
                        history.pop(0)
                    if best_abs <= 1e-8:
                        return best_x
    
    # Phase 2: Manual Brent's method if we have a bracket
    if a is not None and b is not None and fa is not None and fb is not None:
        if (fa > 0) != (fb > 0) and eval_count[0] < max_evals - 20:
            try:
                # Manual Brent's method
                xtol = 1e-14
                rtol = 4 * np.finfo(float).eps
                maxiter = 100
                
                # Ensure f(a) and f(b) have opposite signs
                if fa * fb > 0:
                    a, b = b, a
                    fa, fb = fb, fa
                
                c = a
                fc = fa
                d = b - a
                e = d
                
                for _ in range(maxiter):
                    check_time()
                    if eval_count[0] >= max_evals - 5:
                        break
                    
                    if (fa > 0) == (fb > 0) or abs(fa) < abs(fb):
                        a, b = b, a
                        fa, fb = fb, fa
                        c = a
                        fc = fa
                        e = d
                        d = b - a
                    
                    tol = 2 * np.finfo(float).eps * abs(a) + xtol / 2
                    mid = 0.5 * (b - a)
                    
                    if abs(mid) <= tol or fa == 0:
                        return a
                    
                    if abs(e) >= tol and abs(fa) > abs(fb):
                        # Inverse quadratic interpolation
                        s = fb / fa
                        if a == c:
                            # Linear interpolation
                            p = 2 * mid * s
                            q = 1 - s
                        else:
                            # Inverse quadratic
                            t1 = fb / fc
                            t2 = fa / fc
                            q = t1 * t2
                            r = t2 * (t1 - s)
                            p = s * (2 * mid * q * (q - r) - (b - c) * (r - 1))
                            q = (q - 1) * (r - 1)
                        
                        # If x = b + p/q is acceptable
                        if (p > 0 and q < 0) or (p < 0 and q > 0):
                            p = -p
                        q = 2 * q
                        m = abs(p) < abs(q) and p / q or mid
                        
                        if m < min(a, b) - tol or m > max(a, b) + tol:
                            m = mid
                        else:
                            d = m
                            e = mid
                    else:
                        # Bisection
                        m = mid
                        e = mid
                        d = m
                    
                    x_new = b + d
                    if not np.isfinite(x_new):
                        break
                    
                    fx_new = safe_eval(x_new)
                    abs_new = abs(fx_new)
                    
                    if abs_new < best_abs:
                        best_x = x_new
                        best_f = fx_new
                        best_abs = abs_new
                    
                    if fx_new == 0:
                        return x_new
                    
                    if (fa > 0) == (fx_new > 0):
                        c = b
                        fc = fb
                        a = x_new
                        fa = fx_new
                    else:
                        c = b
                        fc = fb
                        b = x_new
                        fb = fx_new
                    
                    d = b - a
                    e = mid
                    
                    if abs(d) <= tol or best_abs <= 1e-8:
                        break
                
                if best_abs <= 1e-8:
                    return best_x
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish with secant
    if best_abs > 1e-8 and eval_count[0] < max_evals - 5:
        x = best_x
        fx = best_f
        prev_x = None
        prev_f = None
        
        for j in range(50):
            check_time()
            if eval_count[0] >= max_evals - 2:
                break
            if abs(fx) <= 1e-14:
                return x
            
            # Use secant if we have a previous point
            if prev_x is not None and abs(prev_x - x) > 1e-15:
                d_sec = (fx - prev_f) / (x - prev_x)
                if abs(d_sec) > 1e-30:
                    x_new = x - fx / d_sec
                else:
                    break
            else:
                # Finite difference
                h = 1e-10 * max(1.0, abs(x))
                try:
                    fp = safe_eval(x + h)
                    d = (fp - fx) / h
                    if abs(d) > 1e-30:
                        x_new = x - fx / d
                    else:
                        break
                except RuntimeError:
                    break
            
            if not np.isfinite(x_new):
                break
            
            fx_new = safe_eval(x_new)
            
            if abs(fx_new) < abs(fx):
                prev_x = x
                prev_f = fx
                x = x_new
                fx = fx_new
                if abs(fx) < best_abs:
                    best_x = x
                    best_f = fx
                    best_abs = abs(fx)
                if best_abs <= 1e-8:
                    return best_x
            else:
                # Damped step
                x_new = x - 0.5 * (x - prev_x) if prev_x is not None else x - 0.5 * fx
                if not np.isfinite(x_new):
                    break
                fx_new = safe_eval(x_new)
                if abs(fx_new) < abs(fx):
                    prev_x = x
                    prev_f = fx
                    x = x_new
                    fx = fx_new
                    if abs(fx) < best_abs:
                        best_x = x
                        best_f = fx
                        best_abs = abs(fx)
                break
    
    return best_x
# EVOLVE-BLOCK-END
