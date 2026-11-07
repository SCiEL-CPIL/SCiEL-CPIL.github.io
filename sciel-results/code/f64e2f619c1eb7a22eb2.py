import math
import time
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    x0 = float(x0)
    
    # We will count our own calls to ensure we don't exceed the limit
    # and to manage our budget.
    # The evaluator counts calls. We must be careful.
    # Let's assume a hard cap of ~450 calls to be safe, leaving room for brentq internals if needed.
    # But brentq calls f too. So we need to be very conservative.
    # Actually, the task says "For each equation, more than 500 calls... makes that equation a failure".
    # So we have 500 calls total for the whole process of finding one root.
    # Let's aim to use no more than 400 calls in our scanning phase, leaving 100 for brentq.
    # Brentq usually converges in < 20-30 iterations.
    
    MAX_CALLS = 480
    calls = 0
    
    def safe_f(x):
        nonlocal calls
        if calls >= MAX_CALLS:
            raise RuntimeError("Max calls exceeded")
        if time.time() > deadline:
            raise RuntimeError("Time limit exceeded")
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            return val
        except Exception:
            return float('inf')
    
    # Initial evaluation
    try:
        fx0 = safe_f(x0)
    except Exception:
        return x0
    
    if not math.isfinite(fx0):
        # Try to find a finite value nearby
        for dx in [1e-4, -1e-4, 1e-3, -1e-3, 0.01, -0.01, 0.1, -0.1]:
            try:
                fx_try = safe_f(x0 + dx)
                if math.isfinite(fx_try):
                    x0 = x0 + dx
                    fx0 = fx_try
                    break
            except Exception:
                continue
        if not math.isfinite(fx0):
            return x0
            
    if abs(fx0) <= 1e-8:
        return x0
    
    best_x = x0
    best_abs = abs(fx0)
    
    # Strategy:
    # 1. Try to find a bracket using a multi-scale scan.
    # 2. If found, use Brent's method.
    # 3. If not, use Secant/Newton hybrid on the best point.
    
    bracket_a = None
    bracket_b = None
    fa = None
    fb = None
    
    # We will scan in both directions with increasing step sizes.
    # The step sizes are chosen to cover a wide range quickly.
    # Scales: 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 25.0, 125.0
    # This covers from very local to global.
    
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 25.0, 125.0, 625.0]
    
    # We maintain a history of points and their function values to check for sign changes
    # between non-adjacent points in the scan history if needed, but usually adjacent is enough.
    # To be safer, we store the last few points.
    
    prev_x = x0
    prev_f = fx0
    
    # We'll iterate through scales and directions
    for scale in scales:
        if calls >= MAX_CALLS - 50:  # Reserve calls for brentq
            break
        if time.time() > deadline:
            break
            
        for direction in [1, -1]:
            if calls >= MAX_CALLS - 50:
                break
            if time.time() > deadline:
                break
                
            # Scan a few steps in this direction
            # How many steps? Let's say 10 steps of this scale.
            # So we cover [x0, x0 + 10*scale] or [x0, x0 - 10*scale]
            # Total points added per scale/direction: 10
            # Total points per scale: 20
            # Total scales: 9
            # Total points: 180 + 1 (initial) = 181. This is well within budget.
            
            curr_x = prev_x
            curr_f = prev_f
            
            for i in range(1, 11):
                if calls >= MAX_CALLS - 50:
                    break
                if time.time() > deadline:
                    break
                    
                x_try = x0 + direction * i * scale
                try:
                    f_try = safe_f(x_try)
                except Exception:
                    break
                
                if not math.isfinite(f_try):
                    # If we hit a non-finite value, stop scanning in this direction
                    # as the function might be undefined or blowing up
                    break
                
                if abs(f_try) < best_abs:
                    best_abs = abs(f_try)
                    best_x = x_try
                    
                if abs(f_try) <= 1e-8:
                    return x_try
                
                # Check for sign change between curr_x and x_try
                if curr_f * f_try < 0:
                    bracket_a = curr_x
                    bracket_b = x_try
                    fa = curr_f
                    fb = f_try
                    break
                
                # Update current for next step
                curr_x = x_try
                curr_f = f_try
            
            if bracket_a is not None:
                break
        
        if bracket_a is not None:
            break
    
    if bracket_a is not None:
        # Use Brent's method
        try:
            # We need to ensure we don't exceed call limit
            # Brentq uses its own call count internally? No, it calls f.
            # We have reserved 50 calls. Let's hope it's enough.
            # If it fails, we fall back to bisection manually?
            # Or just return the best endpoint?
            
            root = brentq(f, bracket_a, bracket_b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if math.isfinite(root):
                # Verify
                f_root = f(root) # This call is counted by evaluator, but we are near limit?
                # Actually, the evaluator's final check is not counted.
                # But this call IS counted.
                # We should be careful.
                # If we are close to limit, maybe just return root.
                # But if f(root) is not <= 1e-8, we fail.
                # Brentq should guarantee it.
                return root
        except Exception:
            pass
    
    # If no bracket found, or Brent failed, try to refine best_x
    # Use Secant method
    # We need two points. We have x0 and fx0, and best_x and best_abs.
    # If best_x != x0, use them. Else, pick a nearby point.
    
    x1 = best_x
    f1 = best_abs # Wait, we need the sign. We didn't store sign for best_x.
    # Let's re-evaluate best_x to get sign? No, that costs a call.
    # We can store sign during scan.
    
    # Let's restart the scan logic slightly to keep track of best_f (signed)
    # Actually, let's just implement a simple Newton/Secant hybrid from best_x
    # using finite differences.
    
    x = best_x
    try:
        fx = safe_f(x) # Re-eval to be sure? No, we have best_abs but not sign.
        # Let's just do a finite difference Newton step.
        h = 1e-5 * max(1.0, abs(x))
        fx_plus = safe_f(x + h)
        fx_minus = safe_f(x - h)
        
        if math.isfinite(fx_plus) and math.isfinite(fx_minus):
            deriv = (fx_plus - fx_minus) / (2 * h)
            if deriv != 0 and math.isfinite(deriv):
                x_new = x - fx / deriv
                if math.isfinite(x_new):
                    # Check if x_new is better
                    fx_new = safe_f(x_new)
                    if math.isfinite(fx_new) and abs(fx_new) < abs(fx):
                        x = x_new
                        fx = fx_new
    except Exception:
        pass
    
    # One more Newton step if possible
    try:
        h = 1e-5 * max(1.0, abs(x))
        fx_plus = safe_f(x + h)
        fx_minus = safe_f(x - h)
        if math.isfinite(fx_plus) and math.isfinite(fx_minus):
            deriv = (fx_plus - fx_minus) / (2 * h)
            if deriv != 0 and math.isfinite(deriv):
                x_new = x - fx / deriv
                if math.isfinite(x_new):
                    fx_new = safe_f(x_new)
                    if math.isfinite(fx_new) and abs(fx_new) < abs(fx):
                        x = x_new
    except Exception:
        pass
    
    if not math.isfinite(x):
        return x0
        
    return x
# EVOLVE-BLOCK-END
