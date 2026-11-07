"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Robust scalar root finder using:
    1. Inverse Quadratic Interpolation (IQI) if a bracket exists.
    2. Newton's Method on f(x)^2 (minimization of residual) if no bracket (handles even multiplicity).
    3. Bisection as a safety fallback for IQI.
    """
    x = float(x0)
    tol = 1e-8
    max_calls = 450
    
    # State for minimization / no bracket case
    best_x = x
    best_fx = f(x)
    call_count = 1
    
    if abs(best_fx) <= tol:
        return best_x

    # State for bracketing / IQI case
    a = None
    b = None
    fa = None
    fb = None
    
    # 1. Initial Sampling to find a bracket or good local points
    # We sample a few points to see if we can bracket or just find a low residual
    # Points: x0, x0+1, x0-1, x0+2, x0-2, x0+0.5, x0-0.5
    candidates = [x]
    for delta in [0.5, 1.0, 2.0, 4.0, 8.0]:
        candidates.append(x + delta)
        candidates.append(x - delta)
        
    # Evaluate candidates
    points = [] # list of (x, f(x))
    for cx in candidates:
        if call_count >= max_calls:
            break
        cf = f(cx)
        call_count += 1
        points.append((cx, cf))
        if abs(cf) < abs(best_fx):
            best_x = cx
            best_fx = cf
        if abs(cf) <= tol:
            return cx

    # Check for sign change in points to establish bracket
    # Sort by x for easier bracket check? Or just check pairs.
    # Let's sort by x to find adjacent sign changes.
    points.sort(key=lambda p: p[0])
    
    for i in range(len(points) - 1):
        x1, f1 = points[i]
        x2, f2 = points[i+1]
        if f1 * f2 < 0:
            a, fa = x1, f1
            b, fb = x2, f2
            break

    if a is not None:
        # Bracket found. Use IQI + Bisection (Brent-like)
        # We need 3 points for IQI. We have at least 2 (a, b).
        # Let's use the midpoint or a previous point.
        # We will maintain 3 points: p1, p2, p3 with |f(p3)| <= |f(p2)| <= |f(p1)|?
        # Standard Brent's keeps history. Let's simplify.
        
        # Initial 3rd point: midpoint
        m = (a + b) / 2.0
        if call_count < max_calls:
            fm = f(m)
            call_count += 1
            if abs(fm) < abs(best_fx):
                best_x = m
                best_fx = fm
            if abs(fm) <= tol:
                return m
            # Update bracket
            if fm * fa < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        else:
            # If we ran out of calls, return best
            return best_x

        # Now we have a, b, and we can generate IQI steps.
        # We need 3 distinct points for IQI.
        # Let's keep a history of the last 3 points evaluated within the bracket.
        # For simplicity, we will use the current bracket endpoints and the last midpoint/step.
        
        # Let's store points in a list to manage IQI
        # p_prev, p_curr, p_next
        # Actually, let's just use the two endpoints and the point with the smallest |f| so far inside the bracket.
        
        x_min = best_x
        f_min = best_fx
        
        # Ensure x_min is inside [a, b]
        if not (min(a, b) <= x_min <= max(a, b)):
             x_min = (a+b)/2
             f_min = fm # This might be stale if we updated a/b, but fm was calculated. 
             # Actually, if we updated a/b based on fm, x_min is likely m or one of the original points.
             # Let's just re-evaluate if necessary? No, save calls.
             # If x_min is not in bracket, use midpoint.
             # But we already evaluated midpoint.
             
        for i in range(40):
            if call_count >= max_calls:
                break
                
            # Try IQI
            # We need 3 points: (x1, f1), (x2, f2), (x3, f3)
            # Let's use a, b, and x_min (if distinct)
            x1, f1 = a, fa
            x2, f2 = b, fb
            x3, f3 = x_min, f_min
            
            # Check for distinctness and validity
            valid_iqi = False
            if x1 != x2 and x1 != x3 and x2 != x3 and f1 != 0 and f2 != 0 and f3 != 0:
                # Inverse Quadratic Interpolation
                # Formula:
                # x = (x1*x2*f3/(f3-f1)/(f3-f2) + x2*x3*f1/(f1-f2)/(f1-f3) + x3*x1*f2/(f2-f3)/(f2-f1)) / 
                #     (f3/(f3-f1)/(f3-f2) + f1/(f1-f2)/(f1-f3) + f2/(f2-f3)/(f2-f1))
                
                try:
                    d1 = (f3 - f1) * (f3 - f2)
                    d2 = (f1 - f2) * (f1 - f3)
                    d3 = (f2 - f3) * (f2 - f1)
                    
                    if d1 != 0 and d2 != 0 and d3 != 0:
                        term1 = x1 * x2 * f3 / (d1)
                        term2 = x2 * x3 * f1 / (d2)
                        term3 = x3 * x1 * f2 / (d3)
                        
                        num = term1 + term2 + term3
                        den = f3 / (d1) + f1 / (d2) + f2 / (d3)
                        
                        if den != 0:
                            x_new = num / den
                            # Check if x_new is inside bracket
                            if min(a, b) < x_new < max(a, b):
                                valid_iqi = True
                except (ZeroDivisionError, OverflowError):
                    pass
            
            if valid_iqi and call_count < max_calls:
                f_new = f(x_new)
                call_count += 1
                if abs(f_new) < abs(best_fx):
                    best_x = x_new
                    best_fx = f_new
                if abs(f_new) <= tol:
                    return x_new
                
                # Update bracket
                if f_new * fa < 0:
                    b, fb = x_new, f_new
                else:
                    a, fa = x_new, f_new
                    
                # Update x_min if this is better
                if abs(f_new) < abs(f_min):
                    x_min, f_min = x_new, f_new
                    
            else:
                # Fallback to Bisection
                x_new = (a + b) / 2.0
                if call_count >= max_calls:
                    break
                f_new = f(x_new)
                call_count += 1
                if abs(f_new) < abs(best_fx):
                    best_x = x_new
                    best_fx = f_new
                if abs(f_new) <= tol:
                    return x_new
                    
                if f_new * fa < 0:
                    b, fb = x_new, f_new
                else:
                    a, fa = x_new, f_new
                    
                if abs(f_new) < abs(f_min):
                    x_min, f_min = x_new, f_new

        return best_x

    else:
        # No bracket found. Assume even multiplicity or no real root in vicinity?
        # Task says "root finder", implying a root exists.
        # Use Newton on f(x)^2.
        # g(x) = f(x)^2
        # g'(x) = 2 f(x) f'(x)
        # Newton step: x - g/g' = x - f(x)^2 / (2 f(x) f'(x)) = x - f(x) / (2 f'(x))
        
        x_curr = best_x
        f_curr = best_fx
        
        # Estimate derivative
        h = 1e-4 * max(1.0, abs(x_curr))
        
        for i in range(40):
            if call_count >= max_calls:
                break
            
            # Finite difference derivative
            if call_count >= max_calls - 1:
                break
            x_plus = x_curr + h
            f_plus = f(x_plus)
            call_count += 1
            if abs(f_plus) < abs(best_fx):
                best_x = x_plus
                best_fx = f_plus
            if abs(f_plus) <= tol:
                return x_plus
                
            f_minus = f(x_curr - h) # Wait, we need f(x_curr - h) for central diff? Or just forward?
            # We already have f_curr. Let's use central if possible, but it costs 2 calls.
            # Let's use forward diff: f'(x) ~ (f(x+h) - f(x))/h
            # We have f_plus and f_curr.
            fp = (f_plus - f_curr) / h
            
            if abs(fp) < 1e-15:
                # Derivative too small, shrink h
                h *= 0.5
                continue
                
            # Newton step for f^2 minimization
            # x_new = x_curr - f_curr / (2 * fp)
            x_new = x_curr - f_curr / (2.0 * fp)
            
            if not math.isfinite(x_new):
                break
                
            # Damping: if step is too large, damp it
            step = x_new - x_curr
            if abs(step) > 10 * max(1.0, abs(x_curr)):
                x_new = x_curr + math.copysign(1.0, step) * 1.0
            elif abs(step) > 1.0 * max(1.0, abs(x_curr)):
                x_new = x_curr + math.copysign(0.5, step) * 1.0

            if call_count >= max_calls:
                break
            f_new = f(x_new)
            call_count += 1
            
            if abs(f_new) < abs(best_fx):
                best_x = x_new
                best_fx = f_new
            if abs(f_new) <= tol:
                return x_new
                
            # Update state
            x_curr = x_new
            f_curr = f_new
            
            # Adapt h
            if abs(f_new) < abs(f_curr):
                h *= 0.5
            else:
                h *= 2.0
                if h > 10.0: h = 10.0
                if h < 1e-6: h = 1e-6

        return best_x

# EVOLVE-BLOCK-END
