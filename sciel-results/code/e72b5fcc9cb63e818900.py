"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f near x0 using Brent's method if a bracket is found,
    otherwise Secant/IQI to minimize |f(x)|.
    """
    DEADLINE = 150.0
    start_time = time.time()
    
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    
    if best_abs_fx <= 1e-8:
        return best_x

    # Helper to update best
    def update_best(x_val, f_val):
        nonlocal best_x, best_fx, best_abs_fx
        if abs(f_val) < best_abs_fx:
            best_x = x_val
            best_fx = f_val
            best_abs_fx = abs(f_val)
        if best_abs_fx <= 1e-8:
            return True
        return False

    # Phase 1: Try to find a bracket
    a, b = None, None
    fa, fb = None, None
    
    # Start with x0 as center
    xc = x
    fc = best_fx
    update_best(xc, fc)
    
    step = max(1e-4, abs(xc) * 1e-4)
    
    # Expand to find sign change
    for i in range(1, 100):
        if time.time() - start_time > DEADLINE - 10:
            break
        # Right
        xr = xc + step * i
        fr = f(xr)
        update_best(xr, fr)
        if best_abs_fx <= 1e-8:
            return best_x
        if fc * fr < 0:
            a, b = xc, xr
            fa, fb = fc, fr
            break
        # Left
        xl = xc - step * i
        fl = f(xl)
        update_best(xl, fl)
        if best_abs_fx <= 1e-8:
            return best_x
        if fc * fl < 0:
            a, b = xl, xc
            fa, fb = fl, fc
            break
        # Increase step size exponentially to cover larger ranges
        if i == 10:
            step *= 10.0

    # Phase 2: Solve
    if a is not None and b is not None:
        # Use Brent's Method
        # Variables for Brent's:
        # a, b are bracket
        # c is previous b
        # d is previous a
        # fa, fb, fc, fd are values
        # We need to maintain the invariant that a and b are the current bracket
        # and c is the best guess (min |f|) within [a,b]
        
        # Initialize
        if abs(fa) < abs(fb):
            # Swap so that |fa| > |fb| is not required, but usually we track c as the best
            pass
            
        # Standard Brent's setup:
        # c is the best estimate of the root so far
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            
        c = a
        fc = fa
        
        # We need a second point to start Secant/IQI. 
        # We have a, b. Let's use b as the next guess? Or midpoint?
        # Brent's method typically starts with a and b, and computes a new point.
        # Let's use the standard algorithm structure.
        
        # To use IQI, we need 3 points. We have a, b. 
        # Let's generate a third point using Secant first to populate history.
        
        # History for IQI: (x1, f1), (x2, f2), (x3, f3)
        # Let's just use Secant for the first step to get a third point, then IQI.
        
        # Actually, simpler: Use a robust hybrid loop.
        # 1. Try IQI if we have 3 distinct points.
        # 2. Else try Secant if we have 2 distinct points.
        # 3. Else Bisection.
        # 4. Ensure the new point is inside [a, b] if a bracket exists.
        
        x1 = a
        f1 = fa
        x2 = b
        f2 = fb
        
        # We need a 3rd point to start IQI.
        # Let's use the midpoint or secant from x1, x2.
        # Secant from x1, x2:
        if f1 != f2:
            x3 = x1 - f1 * (x2 - x1) / (f2 - f1)
        else:
            x3 = (x1 + x2) / 2.0
            
        # Clip x3 to [a, b]
        if x3 < a or x3 > b:
            x3 = (a + b) / 2.0
            
        f3 = f(x3)
        update_best(x3, f3)
        if best_abs_fx <= 1e-8:
            return best_x

        # Now we have 3 points: x1, x2, x3.
        # We need to maintain a bracket [a, b].
        # Let's reset a, b to be the current bracket containing the root.
        # We know sign change was between original a and b.
        # Let's call them a_bracket, b_bracket.
        a_bracket = a
        b_bracket = b
        fa_bracket = fa
        fb_bracket = fb
        
        # History for IQI
        hist_x = [x1, x2, x3]
        hist_f = [f1, f2, f3]
        
        for it in range(100):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Check if current best is good enough
            if best_abs_fx <= 1e-8:
                return best_x
                
            # Determine new point
            x_new = None
            f_new = None
            
            # Try IQI
            # Check if hist points are distinct enough
            if len(hist_x) >= 3:
                x_a, x_b, x_c = hist_x[-3], hist_x[-2], hist_x[-1]
                f_a, f_b, f_c = hist_f[-3], hist_f[-2], hist_f[-1]
                
                # Inverse Quadratic Interpolation
                # Fit x = A*f^2 + B*f + C
                # Using Lagrange or solving linear system
                # Formula:
                # x_new = (x_a * f_b * f_c / ((f_a-f_b)*(f_a-f_c)) +
                #          x_b * f_a * f_c / ((f_b-f_a)*(f_b-f_c)) +
                #          x_c * f_a * f_b / ((f_c-f_a)*(f_c-f_b)))
                
                denom1 = (f_a - f_b) * (f_a - f_c)
                denom2 = (f_b - f_a) * (f_b - f_c)
                denom3 = (f_c - f_a) * (f_c - f_b)
                
                if abs(denom1) > 1e-15 and abs(denom2) > 1e-15 and abs(denom3) > 1e-15:
                    term1 = x_a * f_b * f_c / denom1
                    term2 = x_b * f_a * f_c / denom2
                    term3 = x_c * f_a * f_b / denom3
                    x_iqi = term1 + term2 + term3
                    
                    # Check if x_iqi is reasonable
                    # It should be within the bracket
                    if a_bracket <= x_iqi <= b_bracket or b_bracket <= x_iqi <= a_bracket:
                        # Also check if it's not too close to existing points
                        if min(abs(x_iqi - x_a), abs(x_iqi - x_b), abs(x_iqi - x_c)) > 1e-15 * max(1.0, abs(x_iqi)):
                            x_new = x_iqi
                            f_new = f(x_new)
                            update_best(x_new, f_new)
                            if best_abs_fx <= 1e-8:
                                return best_x
                            
                            # Update history
                            hist_x.append(x_new)
                            hist_f.append(f_new)
                            if len(hist_x) > 5:
                                hist_x.pop(0)
                                hist_f.pop(0)
                            
                            # Update bracket
                            # If f_new has same sign as fa_bracket, a_bracket = x_new
                            # If f_new has same sign as fb_bracket, b_bracket = x_new
                            if (fa_bracket * f_new) <= 0:
                                b_bracket = x_new
                                fb_bracket = f_new
                            else:
                                a_bracket = x_new
                                fa_bracket = f_new
                                # Ensure a_bracket, b_bracket order is consistent for logic?
                                # It doesn't matter for the sign check, just that they bracket.
                            
                            continue
            # Fallback to Secant
            if len(hist_x) >= 2:
                x_s1, x_s2 = hist_x[-2], hist_x[-1]
                f_s1, f_s2 = hist_f[-2], hist_f[-1]
                if f_s1 != f_s2:
                    x_sec = x_s1 - f_s1 * (x_s2 - x_s1) / (f_s2 - f_s1)
                    # Check if inside bracket
                    if (a_bracket <= x_sec <= b_bracket) or (b_bracket <= x_sec <= a_bracket):
                        if abs(x_sec - x_s1) > 1e-15 * max(1.0, abs(x_sec)):
                            x_new = x_sec
                            f_new = f(x_new)
                            update_best(x_new, f_new)
                            if best_abs_fx <= 1e-8:
                                return best_x
                            
                            hist_x.append(x_new)
                            hist_f.append(f_new)
                            if len(hist_x) > 5:
                                hist_x.pop(0)
                                hist_f.pop(0)
                            
                            if (fa_bracket * f_new) <= 0:
                                b_bracket = x_new
                                fb_bracket = f_new
                            else:
                                a_bracket = x_new
                                fa_bracket = f_new
                            continue
            
            # Fallback to Bisection
            x_mid = (a_bracket + b_bracket) / 2.0
            f_mid = f(x_mid)
            update_best(x_mid, f_mid)
            if best_abs_fx <= 1e-8:
                return best_x
            
            hist_x.append(x_mid)
            hist_f.append(f_mid)
            if len(hist_x) > 5:
                hist_x.pop(0)
                hist_f.pop(0)
                
            if (fa_bracket * f_mid) <= 0:
                b_bracket = x_mid
                fb_bracket = f_mid
            else:
                a_bracket = x_mid
                fa_bracket = f_mid
                
            # Convergence check
            if abs(b_bracket - a_bracket) < 1e-15 * max(1.0, abs(a_bracket), abs(b_bracket)):
                break

    else:
        # No bracket found. Use Secant/Newton to minimize |f(x)|
        # Start with x0 and a perturbed x0
        x1 = x
        f1 = best_fx
        step = max(1e-4, abs(x1) * 1e-4)
        x2 = x1 + step
        f2 = f(x2)
        update_best(x2, f2)
        if best_abs_fx <= 1e-8:
            return best_x
            
        hist_x = [x1, x2]
        hist_f = [f1, f2]
        
        for it in range(100):
            if time.time() - start_time > DEADLINE - 5:
                break
            if best_abs_fx <= 1e-8:
                return best_x
                
            # Try Secant
            if len(hist_x) >= 2:
                x_s1, x_s2 = hist_x[-2], hist_x[-1]
                f_s1, f_s2 = hist_f[-2], hist_f[-1]
                if f_s1 != f_s2:
                    x_new = x_s1 - f_s1 * (x_s2 - x_s1) / (f_s2 - f_s1)
                    if math.isfinite(x_new):
                        f_new = f(x_new)
                        update_best(x_new, f_new)
                        if best_abs_fx <= 1e-8:
                            return best_x
                        hist_x.append(x_new)
                        hist_f.append(f_new)
                        if len(hist_x) > 5:
                            hist_x.pop(0)
                            hist_f.pop(0)
                        continue
                    
            # If Secant fails or is not finite, try small step in direction of negative gradient (approx)
            # Or just step towards 0 if x is large?
            # Let's just try a small step
            x_new = x1 + step
            f_new = f(x_new)
            update_best(x_new, f_new)
            if best_abs_fx <= 1e-8:
                return best_x
            x1 = x_new
            f1 = f_new
            hist_x.append(x_new)
            hist_f.append(f_new)
            if len(hist_x) > 5:
                hist_x.pop(0)
                hist_f.pop(0)
            step *= 0.5
            if step < 1e-15:
                break

    return best_x
# EVOLVE-BLOCK-END
