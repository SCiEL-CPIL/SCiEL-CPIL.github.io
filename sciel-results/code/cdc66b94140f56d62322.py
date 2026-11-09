import time
import math
import numpy as np

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    max_evals = 500
    eval_count = 0
    
    # Best result tracker
    best_x = float(x0)
    best_fx = None
    best_abs_fx = float('inf')
    
    def safe_eval(x):
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if time.time() > deadline or eval_count >= max_evals:
            return None
        eval_count += 1
        try:
            val = f(x)
            val = float(val)
        except Exception:
            val = float('inf')
        
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_abs_fx:
                best_abs_fx = abs_val
                best_x = x
                best_fx = val
            return val
        else:
            return val

    # Initial evaluation
    x = float(x0)
    fx = safe_eval(x)
    if fx is None:
        return best_x
    if abs(fx) <= 1e-8:
        return x
        
    # Generate initial points
    h = 1e-4 * max(1.0, abs(x))
    points = [(x, fx)]
    
    x1 = x + h
    fx1 = safe_eval(x1)
    if fx1 is None:
        return best_x
    if abs(fx1) <= 1e-8:
        return x1
    points.append((x1, fx1))
    
    x2 = x + 2*h
    fx2 = safe_eval(x2)
    if fx2 is None:
        return best_x
    if abs(fx2) <= 1e-8:
        return x2
    points.append((x2, fx2))
    
    # Main loop
    while time.time() < deadline and eval_count < max_evals:
        # Sort points by |f(x)|
        points.sort(key=lambda p: abs(p[1]) if math.isfinite(p[1]) else float('inf'))
        
        # Take top 3
        p0, p1, p2 = points[0], points[1], points[2]
        x0, f0 = p0
        x1, f1 = p1
        x2, f2 = p2
        
        new_x = None
        
        # Try IQI
        try:
            if math.isfinite(f0) and math.isfinite(f1) and math.isfinite(f2):
                # Check if points are distinct enough
                if abs(x1 - x0) > 1e-12 and abs(x2 - x0) > 1e-12 and abs(x2 - x1) > 1e-12:
                    coeffs = np.polyfit([x0, x1, x2], [f0, f1, f2], 2)
                    roots = np.roots(coeffs)
                    # Find real roots
                    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and math.isfinite(r.real)]
                    if real_roots:
                        # Choose the root closest to the best x (x0)
                        # Or within the range?
                        # Let's pick the one closest to x0
                        new_x = min(real_roots, key=lambda r: abs(r - x0))
        except Exception:
            new_x = None
            
        # Fallback: Secant step using best two points
        if new_x is None:
            # Use p0 and p1 (best two)
            if math.isfinite(f0) and math.isfinite(f1) and abs(f1 - f0) > 1e-30:
                new_x = x0 - f0 * (x1 - x0) / (f1 - f0)
            else:
                # If secant fails, try a small step
                new_x = x0 + h * 0.1
        
        # Check if new_x is valid and distinct
        if not math.isfinite(new_x):
            new_x = x0 + h * 0.1
            
        # Ensure new_x is not too close to existing points
        too_close = False
        for px, _ in points:
            if abs(new_x - px) < 1e-12 * max(1.0, abs(px)):
                too_close = True
                break
                
        if too_close:
            # Perturb
            new_x = x0 + h * (1.0 + 0.1 * eval_count)

        fx_new = safe_eval(new_x)
        if fx_new is None:
            break
            
        if abs(fx_new) <= 1e-8:
            return new_x
            
        # Update points
        points.append((new_x, fx_new))
        # Keep only top 3
        points.sort(key=lambda p: abs(p[1]) if math.isfinite(p[1]) else float('inf'))
        points = points[:3]
        
    return best_x
