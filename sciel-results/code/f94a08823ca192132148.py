"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 140  # 140s safety margin (evaluator kills at 150s)
    eval_count = 0
    max_evals = 500
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
    
    # Try to get initial points
    f0 = safe_f(x)
    if abs(f0) <= 1e-8:
        return x
    
    # Get a second point for secant - try multiple step sizes to find a good direction
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = safe_f(x1)
    if abs(f1) <= 1e-8:
        return x1
    
    # If f1 has same sign and similar magnitude, try opposite direction
    if abs(f1) > abs(f0) * 0.5 and f0 * f1 > 0:
        x1_alt = x - h
        f1_alt = safe_f(x1_alt)
        if abs(f1_alt) <= 1e-8:
            return x1_alt
        if abs(f1_alt) < abs(f1):
            x1, f1 = x1_alt, f1_alt
        elif f0 * f1_alt < 0:  # bracket found
            x1, f1 = x1_alt, f1_alt
    
    # Maintain three points for inverse quadratic interpolation
    # (x0, f0), (x1, f1), and we'll get x2
    # Use secant first to get x2
    if f1 != f0:
        x2 = x1 - f1 * (x1 - x) / (f1 - f0)
    else:
        x2 = x1 + h
    
    f2 = safe_f(x2)
    if abs(f2) <= 1e-8:
        return x2
    
    # Now we have three points: (x, f0), (x1, f1), (x2, f2)
    # Use inverse quadratic interpolation for the rest
    for _ in range(485):  # We've used up to 5 evals, have 495 left
        if time.time() > deadline:
            break
        
        # Sort the three points by |f| value to use the best three
        points = [(x, f0), (x1, f1), (x2, f2)]
        points.sort(key=lambda p: abs(p[1]))
        
        # Check if we already have a good enough root
        if abs(points[0][1]) <= 1e-8:
            return points[0][0]
        
        # Use the three points with smallest |f| for IQI
        xa, fa = points[0]
        xb, fb = points[1]
        xc, fc = points[2]
        
        # Inverse quadratic interpolation
        # We want to find x such that f(x) = 0
        # Using Lagrange interpolation on 1/f values
        # Let's use the formula:
        # x_new = xa*fa*(fa-fc)/(fb*(fb-fc)) + xb*fb*(fb-fa)/(fc*(fc-fa)) + xc*fc*(fc-fb)/(fa*(fa-fb))
        # But this can be numerically unstable. Let's use a more stable form.
        
        # Alternative: fit a quadratic through (fa, xa), (fb, xb), (fc, xc)
        # and find where it crosses 0.
        
        # Using the standard IQI formula with careful handling of near-zero values
        try:
            # If any f value is very close to zero, use secant with the other two
            if abs(fa) < 1e-15 or abs(fb) < 1e-15 or abs(fc) < 1e-15:
                # Use secant method with the two non-zero points
                if abs(fa) < 1e-15:
                    # Use xb, fb and xc, fc
                    if fb != fc:
                        x_new = xc - fc * (xc - xb) / (fc - fb)
                    else:
                        x_new = (xb + xc) / 2
                elif abs(fb) < 1e-15:
                    # Use xa, fa and xc, fc
                    if fa != fc:
                        x_new = xc - fc * (xc - xa) / (fc - fa)
                    else:
                        x_new = (xa + xc) / 2
                else:
                    # Use xa, fa and xb, fb
                    if fa != fb:
                        x_new = xb - fb * (xb - xa) / (fb - fa)
                    else:
                        x_new = (xa + xb) / 2
            else:
                # Full IQI
                # Using the formula from Numerical Recipes
                fa2 = fa * fa
                fb2 = fb * fb
                fc2 = fc * fc
                
                # x_new = (xa * fa * (fa - fc) / (fb * (fb - fc)) + 
                #          xb * fb * (fb - fa) / (fc * (fc - fa)) + 
                #          xc * fc * (fc - fb) / (fa * (fa - fb)))
                # But this has issues when values are close. Let's use a different formulation.
                
                # Using Lagrange polynomial in f-space:
                # P(f) = xa * L_a(f) + xb * L_b(f) + xc * L_c(f)
                # where L_a(f) = (f - fb)(f - fc) / ((fa - fb)(fa - fc))
                # etc.
                # We want P(0) = 0, so:
                # x_new = xa * (fb * fc) / ((fa - fb)(fa - fc)) + 
                #         xb * (fa * fc) / ((fb - fa)(fb - fc)) + 
                #         xc * (fa * fb) / ((fc - fa)(fc - fb))
                
                denom_a = (fa - fb) * (fa - fc)
                denom_b = (fb - fa) * (fb - fc)
                denom_c = (fc - fa) * (fc - fb)
                
                if abs(denom_a) < 1e-30 or abs(denom_b) < 1e-30 or abs(denom_c) < 1e-30:
                    # Degenerate case, use secant
                    if fa != fb:
                        x_new = xb - fb * (xb - xa) / (fb - fa)
                    elif fb != fc:
                        x_new = xc - fc * (xc - xb) / (fc - fb)
                    else:
                        x_new = (xa + xb + xc) / 3
                else:
                    x_new = (xa * fb * fc / denom_a + 
                             xb * fa * fc / denom_b + 
                             xc * fa * fb / denom_c)
        except (ZeroDivisionError, OverflowError):
            # Fallback to secant
            if fa != fb:
                x_new = xb - fb * (xb - xa) / (fb - fa)
            elif fb != fc:
                x_new = xc - fc * (xc - xb) / (fc - fb)
            else:
                x_new = (xa + xb + xc) / 3
        
        # Safety: ensure x_new is not too far from current points
        # If x_new is NaN or Inf, fall back to secant
        if not math.isfinite(x_new):
            if fa != fb:
                x_new = xb - fb * (xb - xa) / (fb - fa)
            elif fb != fc:
                x_new = xc - fc * (xc - xb) / (fc - fb)
            else:
                break
        
        f_new = safe_f(x_new)
        
        if abs(f_new) <= 1e-8:
            # Polish: try to get even tighter with a few secant steps
            for polish_iter in range(5):
                if time.time() > deadline:
                    break
                # Use secant with the two closest points
                pts = sorted([(x_new, f_new), (xa, fa), (xb, fb)], key=lambda p: abs(p[1]))
                px1, pf1 = pts[0]
                px2, pf2 = pts[1]
                if abs(pf1) <= 1e-12:
                    break
                if pf1 != pf2:
                    px3 = px2 - pf2 * (px2 - px1) / (pf2 - pf1)
                else:
                    px3 = (px1 + px2) / 2
                if not math.isfinite(px3):
                    break
                pf3 = safe_f(px3)
                if abs(pf3) <= 1e-12:
                    return px3
                if abs(pf3) < abs(f_new):
                    x_new, f_new = px3, pf3
            return x_new
        
        # Update the three points: drop the one with largest |f|
        # New set of points: (xa, fa), (xb, fb), (xc, fc), (x_new, f_new)
        # Keep the three with smallest |f|
        all_points = [(xa, fa), (xb, fb), (xc, fc), (x_new, f_new)]
        all_points.sort(key=lambda p: abs(p[1]))
        
        # We need to keep the three best, but also ensure we have distinct x values
        # for the next IQI step. If two points have the same x, keep the one with smaller |f|.
        kept = []
        for px, pf in all_points:
            if kept and abs(px - kept[-1][0]) < 1e-15 * max(1.0, abs(px)):
                # Same x, keep the one with smaller |f|
                if abs(pf) < abs(kept[-1][1]):
                    kept[-1] = (px, pf)
            else:
                kept.append((px, pf))
        
        # Take the first three
        if len(kept) >= 3:
            x, f0 = kept[0]
            x1, f1 = kept[1]
            x2, f2 = kept[2]
        else:
            # Shouldn't happen, but break if it does
            break
    
    # Final polish: if we have a good candidate, try to tighten it
    if abs(best_fval) <= 1e-6:
        for polish_iter in range(10):
            if time.time() > deadline:
                break
            # Secant step around best_x
            h_polish = 1e-6 * max(1.0, abs(best_x))
            xp = best_x + h_polish
            fp = safe_f(xp)
            if abs(fp) <= 1e-12:
                return xp
            if fp != best_fval:
                x_new_polish = xp - fp * (xp - best_x) / (fp - best_fval)
            else:
                x_new_polish = (best_x + xp) / 2
            if not math.isfinite(x_new_polish):
                break
            f_new_polish = safe_f(x_new_polish)
            if abs(f_new_polish) <= 1e-12:
                return x_new_polish
            if abs(f_new_polish) < abs(best_fval):
                best_x = x_new_polish
                best_fval = f_new_polish
    
    return best_x
# EVOLVE-BLOCK-END
