# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a single-shot cubic interpolation strategy with a minimal Newton fallback.
    """
    DEADLINE = 110.0
    START_TIME = time.time()
    
    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    def is_finite(x):
        return math.isfinite(x)

    def safe_f(x):
        try:
            fx = f(x)
            if is_finite(fx):
                return fx
        except:
            pass
        return None

    # Initial state
    x0 = float(x0)
    fx0 = safe_f(x0)
    
    if fx0 is None:
        return x0
        
    if abs(fx0) <= 1e-8:
        return x0

    # 1. Attempt Single-Shot Cubic Reconstruction
    # We need 4 points to determine a cubic. We already have x0, f(x0).
    # Choose 3 more points. A local window around x0 is usually best for polynomials.
    # Scale step size based on x0 magnitude to handle large/small scales.
    scale = max(1.0, abs(x0))
    h = 1.0 * scale
    
    points = [
        (x0, fx0),
        (x0 - h, None),
        (x0 + h, None),
        (x0 + 2.0 * h, None)
    ]
    
    valid_points = []
    
    # Evaluate remaining points
    for i in range(1, 4):
        if not check_deadline():
            break
        xi = points[i][0]
        fi = safe_f(xi)
        if fi is not None:
            points[i] = (xi, fi)
            valid_points.append(points[i])
    
    # We need at least 4 points for a cubic. If we don't have 4, we can't do the full cubic.
    # However, the prompt implies "fixed smooth polynomial equations". If they are cubic, 4 points suffice.
    # If they are higher order, this might fail, but the prompt says "3 fixed smooth polynomial equations".
    # Usually in these tasks, "polynomial" might mean low degree. If it's degree > 3, this strategy fails.
    # But the directive says "Replace ... with ... single-shot polynomial reconstruction ... cubic coefficients".
    # So we assume we can fit a cubic.
    
    if len(valid_points) == 4 and check_deadline():
        try:
            xs = np.array([p[0] for p in valid_points])
            ys = np.array([p[1] for p in valid_points])
            
            # Fit cubic: p(x) = c3*x^3 + c2*x^2 + c1*x + c0
            # np.polyfit returns highest degree first
            coeffs = np.polyfit(xs, ys, 3)
            
            # Find roots of the cubic
            roots = np.roots(coeffs)
            
            best_root = None
            min_dist = float('inf')
            
            for r in roots:
                if not is_finite(r):
                    continue
                # Check if this root is a good candidate
                # 1. Distance from x0 (preferring local roots)
                dist = abs(r - x0)
                if dist < min_dist:
                    min_dist = dist
                    best_root = float(r)
            
            if best_root is not None:
                # Verify the root
                f_root = safe_f(best_root)
                if f_root is not None and abs(f_root) <= 1e-8:
                    return best_root
                # If the interpolated root is close but not quite there, 
                # we might use it as a starting point for Newton, 
                # but the strategy is "single-shot". 
                # However, if it's very close, a Newton step might fix it.
                # But let's stick to the strategy: if single-shot fails, fallback.
                
        except Exception:
            pass

    # 2. Fallback: Minimal Safeguarded Newton
    # If the cubic strategy didn't yield a valid root, use a robust Newton method.
    # This handles cases where the polynomial is not cubic, or interpolation was unstable.
    
    x = x0
    fx = fx0
    
    # If we have a best_root from cubic that was close, start there?
    # Let's check if we stored it. Actually, let's just restart from x0 or use the best point found during sampling.
    
    # Re-evaluate best point from the 4 samples
    best_x = x0
    best_fx = fx0
    for i in range(1, 4):
        if points[i][1] is not None:
            if abs(points[i][1]) < abs(best_fx):
                best_fx = points[i][1]
                best_x = points[i][0]
                
    x = best_x
    fx = best_fx
    
    if abs(fx) <= 1e-8:
        return x

    h_deriv = 1e-6 * max(1.0, abs(x))
    
    for _ in range(50):
        if not check_deadline():
            break
            
        if abs(fx) <= 1e-8:
            return x
            
        # Central difference derivative
        fx_p = safe_f(x + h_deriv)
        if fx_p is None:
            break
        fx_m = safe_f(x - h_deriv)
        if fx_m is None:
            break
            
        d = (fx_p - fx_m) / (2 * h_deriv)
        
        if d == 0 or not is_finite(d):
            break
            
        step = fx / d
        if not is_finite(step):
            break
            
        # Safeguard: limit step size
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step * (1 if step > 0 else -1)
            
        x_new = x - step
        if not is_finite(x_new):
            break
            
        fx_new = safe_f(x_new)
        if fx_new is None:
            break
            
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            # Step was bad, reduce step size or break
            h_deriv *= 0.5
            # Try a smaller step in the same direction? Or just break to save time.
            # For robustness, let's try halving the step once.
            step_new = fx / d
            if not is_finite(step_new):
                break
            x_try = x - step_new * 0.5
            fx_try = safe_f(x_try)
            if fx_try is not None and abs(fx_try) < abs(fx):
                x = x_try
                fx = fx_try
            else:
                break

    return x
# EVOLVE-BLOCK-END
