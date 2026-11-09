"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0
    
    eval_count = 0
    max_evals = 500
    
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        eval_count += 1
        try:
            val = f(x)
            if not np.isfinite(val):
                return None
            return float(val)
        except Exception:
            return None

    try:
        x0 = float(x0)
    except:
        return x0

    fx0 = safe_f(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= 1e-8:
        return x0

    # Try to get a second point for secant/IQI
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = safe_f(x1)
    
    if fx1 is None:
        x1 = x0 - h
        fx1 = safe_f(x1)
    
    if fx1 is None:
        return x0

    if abs(fx1) <= 1e-8:
        return x1

    # Maintain history of points for IQI
    # We need at least 3 points for IQI, 2 for secant
    history = [(x0, fx0), (x1, fx1)]
    
    best_x = x0 if abs(fx0) < abs(fx1) else x1
    best_f = fx0 if abs(fx0) < abs(fx1) else fx1
    best_abs_f = abs(best_f)

    # Main loop
    for i in range(max_evals - len(history)):
        if time.time() > deadline:
            break
        
        # Find the two points with smallest |f|
        # Sort by abs(f)
        sorted_hist = sorted(history, key=lambda p: abs(p[1]))
        
        p1 = sorted_hist[0]
        p2 = sorted_hist[1]
        
        x1, f1 = p1
        x2, f2 = p2
        
        # Check for convergence
        if abs(f1) <= 1e-8:
            return x1

        # Check for stagnation
        if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
            # Perturb
            h_perturb = 1e-8 * max(1.0, abs(x1))
            x_new = x1 + h_perturb
            f_new = safe_f(x_new)
            if f_new is None:
                x_new = x1 - h_perturb
                f_new = safe_f(x_new)
            if f_new is not None:
                history.append((x_new, f_new))
                if abs(f_new) < best_abs_f:
                    best_x = x_new
                    best_f = f_new
                    best_abs_f = abs(f_new)
                    if best_abs_f <= 1e-8:
                        return best_x
            continue

        # Try Inverse Quadratic Interpolation (IQI) if we have 3 distinct points
        # We need 3 points with distinct x values
        # Let's pick the 3 best points by |f|, ensuring distinct x
        candidates = []
        seen_x = set()
        for px, fx in sorted_hist:
            # Quantize x to avoid floating point issues in set
            key = round(px, 14)
            if key not in seen_x:
                seen_x.add(key)
                candidates.append((px, fx))
            if len(candidates) == 3:
                break
        
        x_new = None
        
        if len(candidates) >= 3:
            xa, fa = candidates[0]
            xb, fb = candidates[1]
            xc, fc = candidates[2]
            
            # IQI formula:
            # x_new = xa * (fb*fc / ((fa-fb)*(fa-fc))) + xb * (fa*fc / ((fb-fa)*(fb-fc))) + xc * (fa*fb / ((fc-fa)*(fc-fb)))
            
            denom1 = (fa - fb) * (fa - fc)
            denom2 = (fb - fa) * (fb - fc)
            denom3 = (fc - fa) * (fc - fb)
            
            if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                try:
                    term_a = xa * (fb * fc) / denom1
                    term_b = xb * (fa * fc) / denom2
                    term_c = xc * (fa * fb) / denom3
                    x_candidate = term_a + term_b + term_c
                    
                    # Sanity check: is it finite?
                    if np.isfinite(x_candidate):
                        x_new = x_candidate
                except Exception:
                    pass
        
        # Fallback to Secant (Linear Interpolation)
        if x_new is None:
            if f1 == f2:
                # Degenerate
                h_perturb = 1e-8 * max(1.0, abs(x1))
                x_new = x1 + h_perturb
            else:
                x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
        
        # Safety: Damp step if too large
        # Ideally stay near the best points
        max_step = 1.0 * (abs(x1) + abs(x2)) + 1.0
        if abs(x_new - x1) > max_step:
            x_new = x1 + max_step * math.copysign(1.0, x_new - x1)
        
        # Check if x_new is too close to existing points
        too_close = False
        for px, _ in history:
            if abs(x_new - px) < 1e-12 * max(1.0, abs(x_new)):
                too_close = True
                break
        
        if too_close:
            # Try a different perturbation
            h_perturb = 1e-6 * max(1.0, abs(x1))
            x_new = x1 + h_perturb
            # Check again
            for px, _ in history:
                if abs(x_new - px) < 1e-12 * max(1.0, abs(x_new)):
                    x_new = x1 - h_perturb
                    break
        
        f_new = safe_f(x_new)
        
        if f_new is None:
            # Fallback: midpoint
            x_new = 0.5 * (x1 + x2)
            f_new = safe_f(x_new)
            if f_new is None:
                break
        
        history.append((x_new, f_new))
        
        if abs(f_new) < best_abs_f:
            best_x = x_new
            best_f = f_new
            best_abs_f = abs(f_new)
            if best_abs_f <= 1e-8:
                return best_x

    if abs(best_f) <= 1e-8:
        return best_x
        
    return best_x
# EVOLVE-BLOCK-END
