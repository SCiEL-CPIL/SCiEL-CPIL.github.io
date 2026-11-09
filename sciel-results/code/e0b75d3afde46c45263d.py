"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 130
    eval_count = 0
    max_evals = 495
    best_x = float(x0)
    best_fval = float('inf')
    
    def safe_f(x):
        nonlocal eval_count, best_x, best_fval
        if time.time() > deadline:
            raise TimeoutError()
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("max evals")
        val = f(x)
        val = float(val)
        if math.isfinite(val):
            if abs(val) < abs(best_fval):
                best_fval = val
                best_x = x
        return val
    
    x = float(x0)
    
    # Initial evaluations
    f0 = safe_f(x)
    if abs(f0) <= 1e-8:
        return x
    
    h = max(1e-8, 1e-4 * max(1.0, abs(x)))
    x1 = x + h
    f1 = safe_f(x1)
    if abs(f1) <= 1e-8:
        return x1
    
    # Get third point via secant
    if f1 != f0:
        x2 = x1 - f1 * (x1 - x) / (f1 - f0)
        if not math.isfinite(x2):
            x2 = x1 + h
    else:
        x2 = x1 + h
    
    f2 = safe_f(x2)
    if abs(f2) <= 1e-8:
        return x2
    
    # Main iteration loop
    # Keep a list of (x, f) pairs, sorted by |f|
    points = [(x, f0), (x1, f1), (x2, f2)]
    points.sort(key=lambda p: abs(p[1]))
    
    for it in range(480):
        if time.time() > deadline:
            break
        
        if len(points) < 2:
            break
        
        # Check for root
        if abs(points[0][1]) <= 1e-8:
            return points[0][0]
        
        # Get the best points
        xa, fa = points[0]
        xb, fb = points[1]
        
        # Compute secant
        if abs(fb - fa) > 1e-15:
            x_secant = xb - fb * (xb - xa) / (fb - fa)
        else:
            x_secant = (xa + xb) * 0.5
        
        # Compute IQI if we have 3 points
        x_iqi = None
        if len(points) >= 3:
            xc, fc = points[2]
            try:
                if (abs(fa - fb) > 1e-15 and abs(fa - fc) > 1e-15 and abs(fb - fc) > 1e-15):
                    denom_a = (fa - fb) * (fa - fc)
                    denom_b = (fb - fa) * (fb - fc)
                    denom_c = (fc - fa) * (fc - fb)
                    
                    if (abs(denom_a) > 1e-30 and abs(denom_b) > 1e-30 and abs(denom_c) > 1e-30):
                        x_iqi = (xa * fb * fc / denom_a + 
                                 xb * fa * fc / denom_b + 
                                 xc * fa * fb / denom_c)
            except (ZeroDivisionError, OverflowError, FloatingPointError):
                pass
        
        # Choose next point
        if x_iqi is not None and math.isfinite(x_iqi):
            # Compare steps
            step_s = abs(x_secant - xa)
            step_i = abs(x_iqi - xa)
            if step_s > 1e-15:
                ratio = step_i / step_s
                # Use IQI if it's not too far off
                if 0.05 < ratio < 20.0:
                    x_new = x_iqi
                else:
                    x_new = x_secant
            else:
                x_new = x_iqi if step_i > 1e-15 else (xa + xb) * 0.5
        else:
            x_new = x_secant
        
        # Safety check
        if not math.isfinite(x_new):
            x_new = (xa + xb) * 0.5
            if not math.isfinite(x_new):
                break
        
        # Ensure x_new is different from existing points
        too_close = False
        for px, pf in points:
            if abs(x_new - px) < 1e-12 * max(1.0, abs(x_new)):
                too_close = True
                break
        if too_close:
            # Try secant if we used IQI, or vice versa
            if x_new == x_iqi:
                x_new = x_secant
            else:
                x_new = xa + (xb - xa) * 1.1
        
        if not math.isfinite(x_new):
            break
        
        f_new = safe_f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Add new point
        points.append((x_new, f_new))
        
        # Sort and keep best 5 points (with distinct x values)
        points.sort(key=lambda p: abs(p[1]))
        kept = []
        for px, pf in points:
            # Check if x is distinct
            dup = False
            for kx, kf in kept:
                if abs(px - kx) < 1e-12 * max(1.0, abs(px)):
                    dup = True
                    break
            if not dup:
                kept.append((px, pf))
            if len(kept) >= 5:
                break
        points = kept
    
    return best_x
# EVOLVE-BLOCK-END
