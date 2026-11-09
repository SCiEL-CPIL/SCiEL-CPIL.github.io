"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 125
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
    
    def is_close_to_existing(x_new, pts, tol=1e-12):
        for px, pf in pts:
            if abs(x_new - px) < tol * max(1.0, abs(x_new)):
                return True
        return False
    
    x = float(x0)
    
    # Initial evaluations with multi-scale probing
    f0 = safe_f(x)
    if abs(f0) <= 1e-8:
        return x
    
    # Probe at multiple scales in both directions to find a sign change
    scales = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    base_h = max(1e-8, 1e-4 * max(1.0, abs(x)))
    
    candidates = []
    for s in scales:
        h = s * base_h
        if h < 1e-14:
            h = 1e-14
        x_plus = x + h
        x_minus = x - h
        candidates.append(x_plus)
        candidates.append(x_minus)
    
    # Try candidates to find a point with sign change or small |f|
    x1, f1 = None, None
    for xc in candidates:
        if not math.isfinite(xc):
            continue
        if is_close_to_existing(xc, [(x, f0)], 1e-13):
            continue
        try:
            fc = safe_f(xc)
        except (TimeoutError, ValueError):
            break
        if not math.isfinite(fc):
            continue
        if abs(fc) <= 1e-8:
            return xc
        if x1 is None:
            x1, f1 = xc, fc
        elif f0 * fc < 0:
            # Found sign change, use this as x1
            x1, f1 = xc, fc
            break
        elif x1 is not None and f1 * fc < 0:
            # Sign change between x1 and xc
            x1, f1 = xc, fc
            break
    
    if x1 is None:
        # Fallback: use first candidate
        x1 = x + base_h
        f1 = safe_f(x1)
        if abs(f1) <= 1e-8:
            return x1
    
    # Get third point via secant from best two
    if abs(f1 - f0) > 1e-15:
        x2 = x1 - f1 * (x1 - x) / (f1 - f0)
        if not math.isfinite(x2) or is_close_to_existing(x2, [(x, f0), (x1, f1)], 1e-13):
            x2 = x1 + base_h
    else:
        x2 = x1 + base_h
    
    f2 = safe_f(x2)
    if abs(f2) <= 1e-8:
        return x2
    
    # Main iteration loop
    points = [(x, f0), (x1, f1), (x2, f2)]
    points.sort(key=lambda p: abs(p[1]))
    
    # Track stagnation
    prev_best = abs(points[0][1])
    stagnation_count = 0
    polish_mode = False
    
    for it in range(480):
        if time.time() > deadline:
            break
        
        if len(points) < 2:
            break
        
        # Check for root
        if abs(points[0][1]) <= 1e-8:
            return points[0][0]
        
        # Detect stagnation
        curr_best = abs(points[0][1])
        if curr_best >= prev_best * 0.99:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = curr_best
        
        # Enter polish mode if converging slowly
        if curr_best < 1e-4 and not polish_mode:
            polish_mode = True
        
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
                        # Validate IQI: should be in a reasonable range
                        if math.isfinite(x_iqi):
                            # Check it's not too far from the bracket
                            span = max(abs(xa - xb), abs(xa - xc), abs(xb - xc), 1e-10)
                            if abs(x_iqi - xa) > 100 * span:
                                x_iqi = None
            except (ZeroDivisionError, OverflowError, FloatingPointError):
                pass
        
        # Choose next point
        if x_iqi is not None and math.isfinite(x_iqi):
            step_s = abs(x_secant - xa)
            step_i = abs(x_iqi - xa)
            if step_s > 1e-15:
                ratio = step_i / step_s
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
        if is_close_to_existing(x_new, points, 1e-12):
            if x_new != x_secant and x_iqi is not None:
                x_new = x_secant
            elif x_new != x_iqi and x_iqi is not None and math.isfinite(x_iqi):
                x_new = x_iqi
            else:
                # Perturb
                x_new = xa + (xb - xa) * 1.1
                if is_close_to_existing(x_new, points, 1e-12):
                    x_new = xa - (xb - xa) * 0.1
                    if is_close_to_existing(x_new, points, 1e-12):
                        # Give up on this iteration
                        continue
        
        if not math.isfinite(x_new):
            break
        
        f_new = safe_f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        if not math.isfinite(f_new):
            continue
        
        # Add new point
        points.append((x_new, f_new))
        
        # Sort and keep best 5 points (with distinct x values)
        points.sort(key=lambda p: abs(p[1]))
        kept = []
        for px, pf in points:
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
        
        # If stagnating, try a different strategy
        if stagnation_count >= 3 and len(points) >= 2:
            # Try bisection between the two best points
            mid = (points[0][0] + points[1][0]) * 0.5
            if not is_close_to_existing(mid, points, 1e-12):
                try:
                    f_mid = safe_f(mid)
                    if abs(f_mid) <= 1e-8:
                        return mid
                    if math.isfinite(f_mid):
                        points.append((mid, f_mid))
                        points.sort(key=lambda p: abs(p[1]))
                        kept = []
                        for px, pf in points:
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
                except (TimeoutError, ValueError):
                    break
            stagnation_count = 0
    
    return best_x
# EVOLVE-BLOCK-END
