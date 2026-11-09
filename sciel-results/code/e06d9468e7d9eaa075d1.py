"""Root finder for 3 fixed oscillatory equations with optimized call budget."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x with |f(x)| <= 1e-8.
    
    Strategy:
    1. Local bracketing search with geometric steps.
    2. Hybrid Secant/Bisection convergence (Illinois-style logic).
    3. Final high-precision polish pass if close.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    tol = 1e-8
    tol_tight = 1e-12  # Tighter tolerance for polishing
    
    x0 = float(x0)
    
    # 1. Initial Evaluation
    try:
        f0 = f(x0)
    except Exception:
        # If f(x0) fails, try a slight offset
        x_try = x0 + 1e-6
        try:
            f0 = f(x_try)
            x0 = x_try
        except Exception:
            return x0  # Last resort
    
    if not math.isfinite(f0):
        return x0
        
    if abs(f0) <= tol:
        return x0

    # 2. Bracketing Search
    # We look for a sign change in [x0-d, x0] or [x0, x0+d]
    # We use a set of geometrically spaced distances
    deltas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    a = None
    b = None
    fa = None
    fb = None
    
    # We also keep track of the best point (smallest |f|) found
    best_x = x0
    best_f = f0
    best_abs = abs(f0)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if math.isfinite(x) and math.isfinite(fx):
            af = abs(fx)
            if af < best_abs:
                best_x = x
                best_f = fx
                best_abs = af

    for d in deltas:
        if time.time() > deadline:
            break
            
        x_left = x0 - d
        x_right = x0 + d
        
        # Evaluate left
        try:
            fl = f(x_left)
        except Exception:
            fl = float('nan')
            
        if math.isfinite(fl):
            update_best(x_left, fl)
            if abs(fl) <= tol:
                return x_left
            if fa is None:
                fa = fl
                a = x_left
            # Check sign change with x0
            if fa * f0 < 0:
                # Found bracket [x_left, x0]
                a, fa = x_left, fl
                b, fb = x0, f0
                break
        
        # Evaluate right
        try:
            fr = f(x_right)
        except Exception:
            fr = float('nan')
            
        if math.isfinite(fr):
            update_best(x_right, fr)
            if abs(fr) <= tol:
                return x_right
            if fb is None:
                fb = fr
                b = x_right
            # Check sign change with x0
            if f0 * fr < 0:
                # Found bracket [x0, x_right]
                a, fb = x0, f0
                b, fb = x_right, fr
                break
        
        # Try Steffensen's method as a fallback when no bracket found yet
        if a is None and b is None:
            steff_x = _steffensen(f, x0, best_abs, tol, deadline, update_best)
            if steff_x is not None:
                return steff_x

    # If we found a bracket, use Hybrid Secant/Bisection
    if a is not None and b is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Hybrid Iteration
        for _ in range(50):
            if time.time() > deadline:
                break
                
            # Check current endpoints
            if abs(fa) <= tol:
                return a
            if abs(fb) <= tol:
                return b
            
            # Calculate Secant Point
            denom = fb - fa
            if denom == 0 or not math.isfinite(denom):
                # Fallback to Bisection
                x_new = (a + b) / 2
            else:
                x_secant = b - fb * (b - a) / denom
                
                # Check if Secant point is inside bracket
                if a < x_secant < b:
                    x_new = x_secant
                else:
                    # If outside, use Bisection
                    x_new = (a + b) / 2
                    
            if not math.isfinite(x_new):
                break
                
            try:
                f_new = f(x_new)
            except Exception:
                break
                
            if not math.isfinite(f_new):
                break
                
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
                
            # Update Bracket
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
                
            # Convergence check on x
            if abs(b - a) < 1e-15 * max(1.0, abs(b)):
                break
        
        # Polish pass: If we are close, try to get even closer using a tighter tolerance
        if best_abs <= 1e-6:
            # Use best_x as center for a very local bisection/secant
            if a is not None and b is not None and min(a, b) <= best_x <= max(a, b):
                pa, pb = a, b
                pfa, pfb = fa, fb
            else:
                # Try to find a tiny bracket around best_x
                eps = 1e-10
                pa, pb = best_x - eps, best_x + eps
                try:
                    pfa = f(pa)
                    pfb = f(pb)
                except Exception:
                    pfa, pfb = float('nan'), float('nan')
                
                if not (math.isfinite(pfa) and math.isfinite(pfb)):
                    return best_x

            if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                for _ in range(25):
                    if time.time() > deadline:
                        break
                    if abs(pfa) <= tol:
                        return pa
                    if abs(pfb) <= tol:
                        return pb
                    
                    # Bisection is safest for polish
                    x_mid = (pa + pb) / 2
                    try:
                        f_mid = f(x_mid)
                    except Exception:
                        break
                    if not math.isfinite(f_mid):
                        break
                    update_best(x_mid, f_mid)
                    if abs(f_mid) <= tol:
                        return x_mid
                    
                    if pfa * f_mid < 0:
                        pb, pfb = x_mid, f_mid
                    else:
                        pa, pfa = x_mid, f_mid

        return best_x if best_abs <= tol else (a + b) / 2

    # No bracket found: try Steffensen's method first
    steff_x = _steffensen(f, x0, best_abs, tol, deadline, update_best)
    if steff_x is not None:
        return steff_x
    
    # Fallback to secant if Steffensen fails
    x1 = x0
    f1 = f0
    
    # Find a second point
    candidates = [(x0 + 1e-2, None), (x0 - 1e-2, None), (x0 + 1.0, None), (x0 - 1.0, None)]
    
    x2 = None
    f2 = None
    min_abs = float('inf')
    
    for cx, _ in candidates:
        try:
            cf = f(cx)
        except Exception:
            continue
        if math.isfinite(cf):
            update_best(cx, cf)
            if abs(cf) < min_abs:
                min_abs = abs(cf)
                x2 = cx
                f2 = cf
    
    if x2 is None:
        # Fallback: try a larger step
        x2 = x0 + 10.0
        try:
            f2 = f(x2)
        except Exception:
            return best_x
        if not math.isfinite(f2):
            return best_x
        update_best(x2, f2)

    # Run Secant
    for _ in range(30):
        if time.time() > deadline:
            break
            
        if abs(f1) <= tol:
            return x1
        if abs(f2) <= tol:
            return x2
        
        denom = f2 - f1
        if denom == 0 or not math.isfinite(denom):
            break
            
        x_new = x2 - f2 * (x2 - x1) / denom
        
        if not math.isfinite(x_new):
            break
            
        try:
            f_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(f_new):
            break
            
        update_best(x_new, f_new)
        
        if abs(f_new) <= tol:
            return x_new
            
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        
    # Polish pass for Secant case
    if best_abs <= 1e-6:
        eps = 1e-10
        pa, pb = best_x - eps, best_x + eps
        try:
            pfa = f(pa)
            pfb = f(pb)
        except Exception:
            return best_x
        
        if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
            for _ in range(25):
                if time.time() > deadline:
                    break
                if abs(pfa) <= tol:
                    return pa
                if abs(pfb) <= tol:
                    return pb
                
                x_mid = (pa + pb) / 2
                try:
                    f_mid = f(x_mid)
                except Exception:
                    break
                if not math.isfinite(f_mid):
                    break
                update_best(x_mid, f_mid)
                if abs(f_mid) <= tol:
                    return x_mid
                
                if pfa * f_mid < 0:
                    pb, pfb = x_mid, f_mid
                else:
                    pa, pfa = x_mid, f_mid

    return best_x


def _steffensen(f, x_start, current_best_abs, tol, deadline, update_best):
    """
    Steffensen's method for root finding.
    Uses the fixed-point iteration x_{n+1} = x_n - f(x_n)/f(x_n + f(x_n))
    with adaptive step sizing to prevent divergence.
    
    Returns x if |f(x)| <= tol, else None.
    """
    x = x_start
    fx = f(x)
    
    if not math.isfinite(fx):
        return None
    
    if abs(fx) <= tol:
        return x
    
    # Adaptive step size based on |f(x)|
    # Start with a small perturbation, adjust based on convergence
    delta = 1e-6 * max(1.0, abs(fx))
    
    for i in range(20):
        if time.time() > deadline:
            return None
        
        # Try the Steffensen step
        x_trial = x + fx
        try:
            fx_trial = f(x_trial)
        except Exception:
            # If evaluation fails, try a smaller step
            delta *= 0.1
            x_trial = x + delta * math.copysign(fx, fx)
            try:
                fx_trial = f(x_trial)
            except Exception:
                return None
        
        if not math.isfinite(fx_trial):
            delta *= 0.1
            continue
        
        # Check if we made progress
        if abs(fx_trial) < abs(fx) * 0.5:
            # Good progress, continue with this step
            x = x_trial
            fx = fx_trial
            update_best(x, fx)
            
            if abs(fx) <= tol:
                return x
            
            # Keep delta similar for next iteration
            delta = abs(fx) * 0.1
        else:
            # No progress or divergence, reduce step size
            delta *= 0.1
            if delta < 1e-12:
                return None
            
            x_trial = x + delta * math.copysign(fx, fx)
            try:
                fx_trial = f(x_trial)
            except Exception:
                return None
            
            if not math.isfinite(fx_trial):
                return None
            
            if abs(fx_trial) < abs(fx):
                x = x_trial
                fx = fx_trial
                update_best(x, fx)
                
                if abs(fx) <= tol:
                    return x
            else:
                # Try negative direction
                x_trial = x - delta * math.copysign(fx, fx)
                try:
                    fx_trial = f(x_trial)
                except Exception:
                    return None
                
                if not math.isfinite(fx_trial):
                    return None
                
                if abs(fx_trial) < abs(fx):
                    x = x_trial
                    fx = fx_trial
                    update_best(x, fx)
                    
                    if abs(fx) <= tol:
                        return x
                else:
                    # No improvement, break
                    break
    
    # Final check
    if abs(fx) <= tol:
        return x
    
    return None

# EVOLVE-BLOCK-END
