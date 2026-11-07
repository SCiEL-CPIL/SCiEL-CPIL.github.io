import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 145.0  # Safety margin (evaluator kills at 150s)
    x0 = float(x0)
    
    call_count = 0
    MAX_CALLS = 490  # Leave buffer for safety
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT

    def safe_f(x):
        nonlocal call_count
        if not time_ok():
            raise TimeoutError("Time limit exceeded")
        if call_count >= MAX_CALLS:
            raise Exception("Call limit exceeded")
        call_count += 1
        val = f(float(x))
        if not np.isfinite(val):
            return float('inf')
        return float(val)

    best_x = x0
    best_f_val = float('inf')
    
    try:
        f0 = safe_f(x0)
        if abs(f0) <= 1e-8:
            return x0
        best_f_val = abs(f0)
        
        # Pure multi-scale bracketing + Brent's method
        # Evaluate f at multiple points per scale to detect sign changes
        # This avoids Newton's failure modes on oscillatory functions
        
        # For each scale, evaluate at x0, x0±s, x0±2s, ... up to a few points
        # Check all adjacent pairs for sign changes, then apply Brent
        
        scales = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        
        for s in scales:
            if not time_ok():
                break
            if call_count >= MAX_CALLS - 30:
                break
            
            # Evaluate at x0 and symmetric points around x0
            # We already have f0 at x0
            points = []
            vals = []
            
            # Add x0 (already evaluated)
            points.append(x0)
            vals.append(f0)
            
            # Evaluate at x0 ± s, x0 ± 2s (if budget allows)
            for mult in [1, 2]:
                if call_count >= MAX_CALLS - 30:
                    break
                for sign in [1, -1]:
                    pt = x0 + sign * mult * s
                    try:
                        v = safe_f(pt)
                    except:
                        continue
                    if np.isfinite(v):
                        points.append(pt)
                        vals.append(v)
                        if abs(v) < best_f_val:
                            best_f_val = abs(v)
                            best_x = pt
                        if abs(v) <= 1e-8:
                            return pt
            
            # Sort points and check adjacent pairs for sign changes
            # Create list of (point, value) pairs
            pv = list(zip(points, vals))
            pv.sort(key=lambda x: x[0])
            
            for i in range(len(pv) - 1):
                if call_count >= MAX_CALLS - 20:
                    break
                p1, v1 = pv[i]
                p2, v2 = pv[i + 1]
                
                if not np.isfinite(v1) or not np.isfinite(v2):
                    continue
                if v1 * v2 < 0:
                    # Sign change detected - use Brent's method
                    a, b = min(p1, p2), max(p1, p2)
                    try:
                        root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=60)
                        if np.isfinite(root):
                            fr = safe_f(root)
                            if np.isfinite(fr) and abs(fr) <= 1e-8:
                                return root
                            elif abs(fr) < best_f_val:
                                best_f_val = abs(fr)
                                best_x = root
                    except:
                        pass
            
            # If we found a very small value, we can stop searching larger scales
            if best_f_val < 1e-10:
                break
        
        # Final polish: if best_x is close to a root, try secant method to refine
        if best_f_val < 1e-4 and call_count < MAX_CALLS - 10:
            # Try a few secant iterations from best_x
            x = best_x
            try:
                fx = safe_f(x)
            except:
                fx = best_f_val
            
            if abs(fx) <= 1e-8:
                return x
            
            # Use a small step for secant
            h = 1e-6 * max(1.0, abs(x))
            for _ in range(5):
                if call_count >= MAX_CALLS - 5 or not time_ok():
                    break
                try:
                    x2 = x + h
                    fx2 = safe_f(x2)
                    if not np.isfinite(fx2):
                        break
                    if abs(fx2) < best_f_val:
                        best_f_val = abs(fx2)
                        best_x = x2
                    if abs(fx2) <= 1e-8:
                        return x2
                    
                    # Secant step
                    if abs(fx2 - fx) > 1e-15:
                        x_new = x - fx * (x2 - x) / (fx2 - fx)
                        if abs(x_new - x) > 10 * max(1.0, abs(x)):
                            break
                        x = x_new
                        try:
                            fx = safe_f(x)
                        except:
                            break
                        if abs(fx) < best_f_val:
                            best_f_val = abs(fx)
                            best_x = x
                        if abs(fx) <= 1e-8:
                            return x
                    else:
                        break
                except:
                    break
        
        return best_x

    except Exception as e:
        return best_x
