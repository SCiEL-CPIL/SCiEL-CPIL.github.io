import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def time_ok():
        return (time.time() - start_time) < deadline

    x = float(x0)
    if not math.isfinite(x):
        x = 0.0

    # First evaluation
    best_x = x
    best_fx = f(x)
    if not math.isfinite(best_fx):
        # Try to find a finite point
        for dx in [1.0, -1.0, 0.1, -0.1, 0.01, -0.01, 10.0, -10.0]:
            x_try = x + dx
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    best_x = x_try
                    best_fx = fx_try
                    break
            except Exception:
                continue
        if not math.isfinite(best_fx):
            return best_x

    best_abs_fx = abs(best_fx)
    if best_abs_fx <= 1e-8:
        return best_x

    # Phase 1: Secant Method (uses previous two points, no derivative eval)
    # We need two points. We have (x0, f(x0)). Let's get another.
    # Use a small step to estimate derivative for the first Newton step, 
    # or just use secant with a perturbed point.
    # To save calls, let's use the secant method directly.
    # x0 = best_x, f0 = best_fx
    # x1 = x0 + h, f1 = f(x1)
    
    h = 1e-4 * max(1.0, abs(best_x))
    x1 = best_x + h
    try:
        f1 = f(x1)
    except Exception:
        f1 = float('nan')
    
    if not math.isfinite(f1):
        # Try other direction
        x1 = best_x - h
        try:
            f1 = f(x1)
        except Exception:
            f1 = float('nan')
            
    if not math.isfinite(f1):
        # Fallback to bracketing immediately
        return _bracket_and_solve(f, best_x, best_fx, time_ok, deadline, start_time)

    # Check if x1 is a root
    if abs(f1) <= 1e-8:
        return x1
    if abs(f1) < best_abs_fx:
        best_x = x1
        best_fx = f1
        best_abs_fx = abs(f1)

    # Secant iterations
    x_prev = best_x
    f_prev = best_fx
    x_curr = x1
    f_curr = f1
    
    # If best_x was updated, adjust x_prev
    if best_x == x1:
        # We don't have a previous point for the new best_x easily without extra calls.
        # Let's just restart secant with (x0, f0) and (x1, f1) regardless of which is "best"
        # Actually, the standard secant method uses the last two points.
        x_prev = best_x - h # Approximate, or just use original x0
        f_prev = best_fx # This is f(x1) if we updated, so this is wrong.
        
    # Let's be careful.
    # Start with p0 = x0, p1 = x1
    p0 = float(x0) if math.isfinite(float(x0)) else 0.0
    fp0 = best_fx # Wait, if best_x moved, best_fx is f(x1).
    # We need to be consistent.
    
    # Let's restart the logic cleanly for Secant.
    # We have evaluated f(x0) and f(x1).
    # Let p0 = x0, fp0 = f(x0)
    # Let p1 = x1, fp1 = f(x1)
    
    # Re-evaluate f(x0) if we overwrote best_fx? No, we stored it.
    # But if x1 was better, best_fx is f(x1).
    # We need f(x0) for the secant step.
    # We didn't store f(x0) separately if we updated best_fx.
    # Let's just use the two points we have: (best_x, best_fx) and the other one.
    
    # To avoid confusion, let's just do a simple Newton-Secant hybrid.
    # Since we have x0 and x1, let's use them as the initial secant pair.
    
    # We need f(x0) and f(x1).
    # We have f(x1) stored in f1.
    # We have f(x0) stored in... well, if best_x == x0, best_fx is f(x0).
    # If best_x == x1, best_fx is f(x1).
    
    # Let's just grab f(x0) again if needed? No, that wastes a call.
    # Let's assume x0 and x1 are the pair.
    # If best_x == x0:
    #   p0 = x0, fp0 = best_fx
    #   p1 = x1, fp1 = f1
    # If best_x == x1:
    #   p0 = x0, fp0 = ? (We lost it if we overwrote best_fx)
    
    # Okay, to be safe and simple:
    # Let's just use the two most recent evaluations.
    # Eval 1: x0, fx0
    # Eval 2: x1, fx1
    
    # If we didn't store fx0 separately, we can't use it if best_fx was overwritten.
    # Let's just store fx0 explicitly.
    
    # Redo Phase 1 start:
    x0_val = float(x0) if math.isfinite(float(x0)) else 0.0
    fx0_val = f(x0_val) # This is a duplicate call if x0_val == best_x initially, but we already called f(x0).
    # Wait, we called f(x) where x=x0. So best_fx IS fx0 initially.
    # Then we called f(x1).
    # So we have fx0 and fx1.
    
    # But if best_x became x1, best_fx is fx1.
    # We still have fx0 in the variable... wait, I didn't keep a separate variable for fx0.
    # I just used best_fx.
    
    # Let's restart the code structure to be cleaner.
    
    return _secant_solve(f, x0_val, fx0_val, x1, f1, best_x, best_fx, time_ok, deadline, start_time)

def _secant_solve(f, x0_val, fx0_val, x1, f1, best_x, best_fx, time_ok, deadline, start_time):
    p0 = x0_val
    fp0 = fx0_val
    p1 = x1
    fp1 = f1
    
    # Ensure p0 != p1
    if abs(p1 - p0) < 1e-12:
        p1 = p0 + 1e-4
        try:
            fp1 = f(p1)
        except Exception:
            fp1 = float('nan')
        if not math.isfinite(fp1):
            return _bracket_and_solve(f, best_x, best_fx, time_ok, deadline, start_time)

    # Secant loop
    for i in range(50):
        if not time_ok():
            return best_x
            
        if abs(fp1) < abs(fp0):
            best_x = p1
            best_fx = fp1
        if abs(fp1) <= 1e-8:
            return p1
            
        # Check for sign change
        if fp0 * fp1 <= 0:
            try:
                root = brentq(f, p0, p1, xtol=1e-12, rtol=1e-12, maxiter=50)
                if math.isfinite(root) and abs(f(root)) <= 1e-8:
                    return root
            except Exception:
                pass

        # Secant step
        denom = fp1 - fp0
        if abs(denom) < 1e-15:
            break
            
        p2 = p1 - fp1 * (p1 - p0) / denom
        
        if not math.isfinite(p2):
            break
            
        # Damping if step is too large
        if abs(p2 - p1) > 10 * max(1.0, abs(p1)):
            p2 = p1 + 0.5 * (p2 - p1) # Or just break
            
        try:
            fp2 = f(p2)
        except Exception:
            break
            
        if not math.isfinite(fp2):
            break
            
        if abs(fp2) < abs(fp1):
            best_x = p2
            best_fx = fp2
        if abs(fp2) <= 1e-8:
            return p2
            
        # Update points
        # Keep the point with smaller |f| as p1? Or just shift?
        # Standard secant shifts. But for robustness, keep the "better" one?
        # Usually shifting is fine.
        p0, fp0 = p1, fp1
        p1, fp1 = p2, fp2
        
        # Convergence check
        if abs(p2 - p1) < 1e-12 * max(1.0, abs(p2)):
             # We converged, but maybe not to 1e-8 accuracy?
             # Check f(p2)
             if abs(fp2) <= 1e-8:
                 return p2
             # If not, try brentq on last interval if sign change exists?
             # Or just return best_x
             pass
             
    # If secant failed, try bracketing
    return _bracket_and_solve(f, best_x, best_fx, time_ok, deadline, start_time)

def _bracket_and_solve(f, center, f_center, time_ok, deadline, start_time):
    # Expand to find a bracket
    step = 0.1
    max_step = 1e6 * max(1.0, abs(center))
    
    # Check immediate neighbors first
    a = center - step
    b = center + step
    
    try:
        fa = f(a)
        fb = f(b)
    except Exception:
        return center
        
    if not math.isfinite(fa) or not math.isfinite(fb):
        # Try to find finite neighbors
        for dx in [0.01, 0.1, 1.0, 10.0]:
            try:
                fa = f(center - dx)
                if math.isfinite(fa):
                    a = center - dx
                    break
            except:
                pass
            try:
                fb = f(center + dx)
                if math.isfinite(fb):
                    b = center + dx
                    break
            except:
                pass
        if not math.isfinite(fa) or not math.isfinite(fb):
            return center

    while fa * fb > 0 and step < max_step:
        if not time_ok():
            return center
            
        step *= 2.0
        a = center - step
        b = center + step
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            break
            
        if not math.isfinite(fa) or not math.isfinite(fb):
            break

    if fa * fb <= 0 and math.isfinite(fa) and math.isfinite(fb):
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root) and abs(f(root)) <= 1e-8:
                return root
        except Exception:
            pass
            
    return center
