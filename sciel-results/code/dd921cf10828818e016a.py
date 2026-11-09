import math
import time
import numpy as np

def find_root(f, x0):
    """
    Finds a root of f starting near x0.
    Strategy: Hybrid Inverse Quadratic Interpolation (IQI) + Secant.
    1. Start with a small local perturbation to get two points.
    2. Use IQI to predict the next step if the function values allow it (smooth, no sign change yet or small sign change).
    3. If IQI fails (division by zero, non-finite, or step too large), fall back to Secant method.
    4. If the method diverges or gets stuck, attempt a local bracket search around the current best point.
    5. If a bracket is found, use Brent's method (via scipy) for guaranteed convergence.
    6. If no bracket is found, return the best point found.
    
    This approach aims for cubic convergence near the root, significantly reducing 
    the number of evaluations compared to Brent's method (which is superlinear but 
    often uses more evaluations for the same tolerance).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    eval_count = 1
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    if math.isnan(fx) or math.isinf(fx):
        # If x0 is bad, try a small step
        x = x + 1e-4 * max(1.0, abs(x))
        fx = f(x)
        eval_count += 1
        best_x = x
        best_fx = fx
        best_abs_fx = abs(fx)
        if math.isnan(fx) or math.isinf(fx):
            return x  # Can't do much if f is invalid everywhere

    if best_abs_fx <= 1e-8:
        return best_x

    # We need a second point to start the iterative method
    # Choose a small step based on the scale of x
    dx = 1e-6 * max(1.0, abs(x))
    x1 = x
    f1 = fx
    x2 = x + dx
    f2 = f(x2)
    eval_count += 1
    
    if math.isnan(f2) or math.isinf(f2):
        # Try other side
        x2 = x - dx
        f2 = f(x2)
        eval_count += 1
        if math.isnan(f2) or math.isinf(f2):
            # Fallback to bracketing immediately
            return _fallback_bracket(f, x, best_x, best_abs_fx, eval_count, start_time, deadline)

    if abs(f2) < best_abs_fx:
        best_x = x2
        best_fx = f2
        best_abs_fx = abs(f2)
        # Swap so x2 is the "new" one, x1 is the "old" one? 
        # Actually, let's maintain x1, x2 as the last two points.
        # If f2 is better, we might want to swap roles to keep x1 as the "older" one?
        # No, just keep x1, x2. The method uses x1, x2 to predict x3.
        
    if best_abs_fx <= 1e-8:
        return best_x

    # Main Loop
    # We will maintain x1, f1 and x2, f2 as the two most recent points.
    # We want to find x3.
    
    max_iter = 100  # Limit iterations to prevent infinite loops if it oscillates
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check for sign change in current segment
        if f1 * f2 < 0:
            # We have a bracket. Use Brent's method for robustness.
            # Brent is safe and fast for bracketed intervals.
            try:
                from scipy.optimize import brentq
                root = brentq(f, x1, x2, maxiter=50, rtol=1e-12, xtol=1e-12)
                fr = f(root)
                if abs(fr) <= 1e-8:
                    return root
                if abs(fr) < best_abs_fx:
                    best_x = root
                    best_abs_fx = abs(fr)
            except Exception:
                pass
            
        # Try IQI
        x3 = None
        # IQI formula:
        # x3 = x1 + (x1 - x2) * f1 * (f1 - f2) / ((f1 - f2)^2 - (f1 - f3_prev) * (f2 - f1)) -- wait, standard IQI uses 3 points.
        # We only have 2 points. We can use Secant first to get a 3rd, or use a simplified IQI?
        # Actually, with 2 points, we can only do Secant.
        # To get IQI, we need 3 points. 
        # Let's modify strategy: 
        # 1. Use Secant to get x3.
        # 2. Then use IQI with (x1, x2, x3) to get x4.
        # This is a standard hybrid.
        
        # Step 1: Secant
        if f2 - f1 == 0:
            # Degenerate, perturb
            x2 += 1e-10 * max(1.0, abs(x2))
            f2 = f(x2)
            eval_count += 1
            continue
            
        x_secant = x2 - f2 * (x2 - x1) / (f2 - f1)
        
        # Step 2: IQI Prediction (using x1, x2, and a hypothetical x3? No, we need 3 actual points)
        # Let's just use Secant for the first few steps, then switch to IQI once we have 3 points.
        
        # Actually, a simpler high-order method with 2 points is just Secant. 
        # But the prompt mentions IQI improves score. 
        # To use IQI, we need to store history.
        
        # Let's restart the loop logic to maintain a history of 3 points.
        # For simplicity and robustness in this single function:
        # Use Secant. If it converges slowly, it will take more evals.
        # Is Secant "structurally different" enough? 
        # The current program uses Brent. Brent is robust but can be slow (many evals).
        # Secant/IQI is faster (fewer evals) if it doesn't diverge.
        
        # Let's implement a proper IQI loop.
        # We need x0, x1, x2.
        # Let's re-initialize the state for the loop to hold 3 points.
        break  # Exit this loop structure and use a better one below

    # --- Refined Strategy: IQI with History ---
    # Re-initialize to manage 3 points properly
    # We already have x (x0), x2 (x0+dx). Let's call them x_1, x_2.
    # We need a third point x_3.
    
    x_1 = x
    f_1 = fx
    x_2 = x + 1e-6 * max(1.0, abs(x))
    f_2 = f(x_2)
    eval_count += 1
    
    if math.isnan(f_2) or math.isinf(f_2):
        return _fallback_bracket(f, x, best_x, best_abs_fx, eval_count, start_time, deadline)
    
    # Get third point via Secant from x_1, x_2
    if f_2 - f_1 != 0:
        x_3 = x_2 - f_2 * (x_2 - x_1) / (f_2 - f_1)
    else:
        x_3 = x_2 + 1e-4
    
    f_3 = f(x_3)
    eval_count += 1
    
    if math.isnan(f_3) or math.isinf(f_3):
        # Fallback
        return _fallback_bracket(f, x, best_x, best_abs_fx, eval_count, start_time, deadline)
    
    # Update best
    if abs(f_3) < best_abs_fx:
        best_x = x_3
        best_abs_fx = abs(f_3)
        best_fx = f_3
    if abs(f_3) <= 1e-8:
        return x_3
        
    # Now we have (x_1, f_1), (x_2, f_2), (x_3, f_3)
    # We will iterate. At each step, we try IQI to get x_new.
    # If IQI fails, use Secant on the two closest points.
    # We keep a window of the last 3 points.
    
    # Sort points by x? No, IQI formula doesn't strictly require sorting, 
    # but it's more stable if we use the three most recent in time or space.
    # Standard IQI uses the three most recent points.
    
    pts = [(x_1, f_1), (x_2, f_2), (x_3, f_3)]
    
    for _ in range(50):  # 50 iterations of high-order method
        if time.time() > deadline:
            break
            
        # Check if any point is a root
        for x_i, f_i in pts:
            if abs(f_i) <= 1e-8:
                return x_i
            if abs(f_i) < best_abs_fx:
                best_x = x_i
                best_abs_fx = abs(f_i)
                best_fx = f_i
                
        # Check for bracket in the last 3 points
        # If (x1,x2) or (x2,x3) or (x1,x3) brackets, use Brent
        # Specifically, check consecutive points
        if pts[0][1] * pts[1][1] < 0:
             return _try_brent(f, pts[0][0], pts[1][0], best_x, best_abs_fx, eval_count, start_time, deadline)
        if pts[1][1] * pts[2][1] < 0:
             return _try_brent(f, pts[1][0], pts[2][0], best_x, best_abs_fx, eval_count, start_time, deadline)
             
        # Try IQI
        x_new = None
        try:
            x1, f1 = pts[0]
            x2, f2 = pts[1]
            x3, f3 = pts[2]
            
            # Inverse Quadratic Interpolation
            # f(x) approximated by quadratic in x? No, x approximated by quadratic in f.
            # x = A f^2 + B f + C
            # We solve for x_new such that f(x_new) = 0.
            # Formula:
            # x_new = x1 + (x1 - x2) * f1 * (f1 - f3) / ((f1 - f2)*(f1 - f3) - (f2 - f1)*(f2 - f3)) 
            # Wait, standard formula for IQI root:
            # x_new = (x1*f2*f3*(f2-f3) + x2*f1*f3*(f3-f1) + x3*f1*f2*(f1-f2)) / 
            #         (f1*f2*(f2-f3) + f2*f3*(f3-f1) + f3*f1*(f1-f2))
            
            denom = f1*f2*(f2-f3) + f2*f3*(f3-f1) + f3*f1*(f1-f2)
            if abs(denom) > 1e-300:
                numer = x1*f2*f3*(f2-f3) + x2*f1*f3*(f3-f1) + x3*f1*f2*(f1-f2)
                x_new = numer / denom
            else:
                x_new = None
        except Exception:
            x_new = None
            
        # Fallback to Secant if IQI fails
        if x_new is None or not math.isfinite(x_new):
            # Use Secant on the two points with smallest |f|
            # Or just the last two
            if abs(f3) < abs(f1):
                # Use x2, x3
                if f3 - f2 != 0:
                    x_new = x3 - f3 * (x3 - x2) / (f3 - f2)
                else:
                    x_new = x3 + 1e-4
            else:
                # Use x1, x2
                if f2 - f1 != 0:
                    x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                else:
                    x_new = x2 + 1e-4
                    
        # Sanity check on step size
        # If step is too large, dampen it
        if not math.isfinite(x_new):
            break
            
        # Heuristic: if x_new is very far from current best, limit step
        # Distance from center of current window
        center = (x1 + x2 + x3) / 3.0
        max_step = 10.0 * abs(center) + 10.0
        if abs(x_new - center) > max_step:
             # Dampen
             x_new = center + max_step * math.copysign(1.0, x_new - center)
             
        f_new = f(x_new)
        eval_count += 1
        
        if math.isnan(f_new) or math.isinf(f_new):
             # Perturb and retry
             x_new = x3 + 1e-5 * max(1.0, abs(x3))
             f_new = f(x_new)
             eval_count += 1
             if math.isnan(f_new) or math.isinf(f_new):
                 break
                 
        # Update best
        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_abs_fx = abs(f_new)
            best_fx = f_new
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update history: drop the oldest point (x1), add new point
        # We want to keep the 3 most recent points in time? 
        # Or the 3 points that define the local geometry?
        # Usually, dropping the oldest is standard for secant/IQI.
        pts = [pts[1], pts[2], (x_new, f_new)]
        
    # If loop finishes without success, try bracketing fallback
    return _fallback_bracket(f, best_x, best_x, best_abs_fx, eval_count, start_time, deadline)

def _try_brent(f, a, b, best_x, best_abs_fx, eval_count, start_time, deadline):
    """
    Attempts to refine the root using Brent's method if a bracket is found.
    """
    if time.time() > deadline:
        return best_x
        
    try:
        from scipy.optimize import brentq
        # Use tight tolerances
        root = brentq(f, a, b, maxiter=50, rtol=1e-12, xtol=1e-12)
        fr = f(root)
        if abs(fr) <= 1e-8:
            return root
        if abs(fr) < best_abs_fx:
            return root
    except Exception:
        pass
    return best_x

def _fallback_bracket(f, x_start, best_x, best_abs_fx, eval_count, start_time, deadline):
    """
    Fallback strategy if high-order methods fail.
    Expanding bracket search around x_start.
    """
    if time.time() > deadline:
        return best_x
        
    x = x_start
    d = 1e-3 * max(1.0, abs(x))
    
    # Try a few expansions
    for i in range(20):
        if time.time() > deadline:
            break
            
        a = x - d
        b = x + d
        
        fa = f(a)
        fb = f(b)
        eval_count += 2
        
        if math.isnan(fa) or math.isinf(fa) or math.isnan(fb) or math.isinf(fb):
            d *= 2.0
            continue
            
        if abs(fa) < best_abs_fx:
            best_x = a
            best_abs_fx = abs(fa)
        if abs(fb) < best_abs_fx:
            best_x = b
            best_abs_fx = abs(fb)
            
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
            
        if fa * fb < 0:
            # Bracket found
            return _try_brent(f, a, b, best_x, best_abs_fx, eval_count, start_time, deadline)
            
        d *= 2.0
        if d > 1e100:
            break
            
    return best_x
