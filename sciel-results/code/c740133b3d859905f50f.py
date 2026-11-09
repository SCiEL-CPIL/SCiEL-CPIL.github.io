"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    # 150s limit, keep 10s safety margin for overhead and returning
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

    # Initialize
    try:
        x0 = float(x0)
    except:
        return x0

    fx0 = safe_f(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= 1e-8:
        return x0

    # Data structures for PLI (Piecewise Linear Interpolation)
    # We maintain a list of (x, f(x)) pairs
    points = [(x0, fx0)]
    
    best_x = x0
    best_f = fx0
    best_abs_f = abs(fx0)

    # Initial exploration to find a second point
    # Try small steps in both directions to get a sense of the function
    h = 1e-4 * max(1.0, abs(x0))
    
    # Try x0 + h
    x1 = x0 + h
    fx1 = safe_f(x1)
    if fx1 is not None:
        points.append((x1, fx1))
        if abs(fx1) < best_abs_f:
            best_x = x1
            best_f = fx1
            best_abs_f = abs(fx1)
            if best_abs_f <= 1e-8:
                return best_x
    else:
        # If right side fails, try left
        x1 = x0 - h
        fx1 = safe_f(x1)
        if fx1 is not None:
            points.append((x1, fx1))
            if abs(fx1) < best_abs_f:
                best_x = x1
                best_f = fx1
                best_abs_f = abs(fx1)
                if best_abs_f <= 1e-8:
                    return best_x
        else:
            # If both fail, we can't do much better than x0
            return best_x

    # Main Loop: PLI
    # Strategy:
    # 1. Find the two points with the smallest |f(x)|.
    # 2. If they have opposite signs, use linear interpolation (Secant-like) to estimate root.
    # 3. If they have same sign, use inverse quadratic interpolation (IQI) if possible, 
    #    or a smaller step towards the better point.
    #    Actually, for robustness on ill-conditioned problems, if signs are same, 
    #    we should expand the search or try a smaller step towards the better point.
    #    However, the prompt suggests PLI specifically. 
    #    Let's stick to:
    #    - If signs differ: Interpolate between them.
    #    - If signs same: Move towards the better point by a factor (e.g., 0.5) or try IQI.
    #    Given "ill-conditioned", IQI can be unstable. 
    #    Let's use a hybrid: 
    #    - Keep track of the last 3 points.
    #    - If we have a bracket (sign change), use Brent's or Secant.
    #    - If no bracket, use the point with min |f| and step towards it.
    
    # To keep it simple and robust as per the directive "PLI model that maintains a sorted list... and performs root-finding via linear interpolation between the two points with the smallest absolute function values":
    # This specific strategy implies we always pick the two "best" points.
    
    for i in range(max_evals - len(points)):
        if time.time() > deadline:
            break
            
        # Find the two points with smallest absolute function values
        # Sort points by abs(f)
        sorted_points = sorted(points, key=lambda p: abs(p[1]))
        
        # If the best point is already good enough, we might have returned, 
        # but let's double check if we haven't.
        if abs(sorted_points[0][1]) <= 1e-8:
            return sorted_points[0][0]
            
        p1 = sorted_points[0]
        p2 = sorted_points[1]
        
        x1, f1 = p1
        x2, f2 = p2
        
        # Check for convergence or stagnation
        if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
            # Points are too close, need to expand
            # Try a step from the best point
            direction = 1.0 if f1 > 0 else -1.0
            # Actually, if they are close, the derivative estimate is bad.
            # Let's try a small perturbation from the best point
            h_perturb = 1e-8 * max(1.0, abs(x1))
            x_new = x1 + h_perturb
            f_new = safe_f(x_new)
            if f_new is None:
                x_new = x1 - h_perturb
                f_new = safe_f(x_new)
            if f_new is not None:
                points.append((x_new, f_new))
                if abs(f_new) < best_abs_f:
                    best_x = x_new
                    best_f = f_new
                    best_abs_f = abs(f_new)
            continue
            
        # Linear Interpolation (Secant step)
        # x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
        if f1 == f2:
            # Degenerate case, perturb
            h_perturb = 1e-8 * max(1.0, abs(x1))
            x_new = x1 + h_perturb
            f_new = safe_f(x_new)
            if f_new is not None:
                points.append((x_new, f_new))
                if abs(f_new) < best_abs_f:
                    best_x = x_new
                    best_f = f_new
                    best_abs_f = abs(f_new)
            continue

        x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
        
        # Safety: If the new point is far away, damp it
        # We want to stay in the neighborhood of the best points
        max_step = 0.5 * (abs(x1) + abs(x2)) + 1.0
        if abs(x_new - x1) > max_step:
            x_new = x1 + max_step * math.copysign(1.0, x_new - x1)
            
        f_new = safe_f(x_new)
        
        if f_new is None:
            # If evaluation fails, try a smaller step towards the better point
            x_new = x1 + 0.5 * (x2 - x1)
            f_new = safe_f(x_new)
            if f_new is None:
                # Try expanding
                x_new = x1 + 1e-4 * max(1.0, abs(x1))
                f_new = safe_f(x_new)
                if f_new is None:
                    break

        points.append((x_new, f_new))
        
        if abs(f_new) < best_abs_f:
            best_x = x_new
            best_f = f_new
            best_abs_f = abs(f_new)
            if best_abs_f <= 1e-8:
                return best_x

        # Optional: If we have a bracket, we could switch to Brent for faster convergence
        # Check if any pair in points forms a bracket
        # This is O(N^2) but N is small (<=500). 
        # To be efficient, we can just check if the new point brackets with any existing point.
        # Actually, for ill-conditioned problems, PLI is often more robust than switching methods prematurely.
        # However, if we find a bracket, Brent is very reliable.
        
        # Let's check for a bracket involving the new point
        bracketed = False
        for px, fx in points:
            if (f_new - fx) * (f1 - fx) < 0 or (f_new - fx) * (f2 - fx) < 0:
                # Found a bracket
                # We could break and use Brent, but let's see if the current PLI step is good.
                # The directive says "Replace the local step-prediction logic with a global PLI model".
                # So we stick to PLI.
                pass

    # Final check
    if abs(best_f) <= 1e-8:
        return best_x
        
    # If we didn't find a root, return the best approximation
    return best_x
# EVOLVE-BLOCK-END
