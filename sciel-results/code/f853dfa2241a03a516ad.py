"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 140  # Safety margin before 150s kill
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    
    def safe_f(x):
        nonlocal eval_count
        if time.time() > deadline:
            raise TimeoutError()
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("max evals exceeded")
        val = f(x)
        if not math.isfinite(val):
            return val
        return float(val)
    
    x = float(x0)
    fx = safe_f(x)
    if abs(fx) <= tol:
        return x
    
    # Maintain a list of (x, fx) sorted by x
    points = [(x, fx)]
    
    def add_point(x_val, fx_val):
        """Add a point to the sorted list, avoiding duplicates."""
        # Check if too close to existing points
        for px, _ in points:
            if abs(px - x_val) < 1e-14 * max(1.0, abs(x_val)):
                return
        # Insert in sorted order
        i = 0
        while i < len(points) and points[i][0] < x_val:
            i += 1
        points.insert(i, (x_val, fx_val))
    
    def get_best_pair():
        """Find the two points with smallest |f| that bracket a sign change,
        or the two closest points if no sign change exists."""
        if len(points) < 2:
            return None
        
        # First try to find a sign change bracket with smallest |f| values
        best_bracket = None
        best_bracket_score = float('inf')
        for i in range(len(points) - 1):
            x1, f1 = points[i]
            x2, f2 = points[i + 1]
            if f1 * f2 <= 0 and math.isfinite(f1) and math.isfinite(f2):
                score = max(abs(f1), abs(f2))
                if score < best_bracket_score:
                    best_bracket_score = score
                    best_bracket = (x1, f1, x2, f2)
        
        if best_bracket:
            return best_bracket
        
        # No sign change: find two points with smallest |f|
        finite_points = [(x, fx) for x, fx in points if math.isfinite(fx)]
        if len(finite_points) < 2:
            return None
        finite_points.sort(key=lambda p: abs(p[1]))
        return (finite_points[0][0], finite_points[0][1], 
                finite_points[1][0], finite_points[1][1])
    
    def interpolate(x1, f1, x2, f2):
        """Linear interpolation to estimate root between (x1,f1) and (x2,f2)."""
        if f1 == f2:
            return (x1 + x2) / 2.0
        # Root of line through (x1,f1) and (x2,f2)
        return x1 - f1 * (x2 - x1) / (f2 - f1)
    
    def bisection(x1, f1, x2, f2):
        return (x1 + x2) / 2.0
    
    # Initial bracketing: try to expand around x0
    h = 1e-3 * max(1.0, abs(x))
    for attempt in range(10):
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
        
        x_left = x - h
        x_right = x + h
        fl = safe_f(x_left)
        fr = safe_f(x_right)
        add_point(x_left, fl)
        add_point(x_right, fr)
        
        if abs(fl) <= tol:
            return x_left
        if abs(fr) <= tol:
            return x_right
        
        if fl * fr < 0:
            break  # Found a bracket
        
        h *= 2.0
        if h > 1e10:
            break
    
    # Main loop: use piecewise linear interpolation
    for _ in range(max_evals):
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
        
        pair = get_best_pair()
        if pair is None:
            break
        
        x1, f1, x2, f2 = pair
        
        # Check if either endpoint is good enough
        if abs(f1) <= tol:
            return x1
        if abs(f2) <= tol:
            return x2
        
        # Determine next point
        has_bracket = (f1 * f2 < 0)
        
        if has_bracket:
            # Use interpolation if it stays within the bracket, else bisection
            x_new = interpolate(x1, f1, x2, f2)
            # Ensure x_new is strictly inside the bracket
            if x1 < x2:
                if x_new <= x1 or x_new >= x2:
                    x_new = bisection(x1, f1, x2, f2)
            else:
                if x_new >= x1 or x_new <= x2:
                    x_new = bisection(x1, f1, x2, f2)
        else:
            # No bracket: use interpolation, but check it's reasonable
            x_new = interpolate(x1, f1, x2, f2)
            # If interpolation goes too far, use a smaller step
            dist = abs(x2 - x1)
            if abs(x_new - x1) > 10 * dist or abs(x_new - x2) > 10 * dist:
                x_new = bisection(x1, f1, x2, f2)
        
        # Check if x_new is too close to existing points
        too_close = False
        for px, _ in points:
            if abs(px - x_new) < 1e-13 * max(1.0, abs(x_new)):
                too_close = True
                break
        if too_close:
            # Nudge slightly
            x_new = x_new + 1e-10 * max(1.0, abs(x_new))
        
        f_new = safe_f(x_new)
        add_point(x_new, f_new)
        
        if abs(f_new) <= tol:
            return x_new
    
    # Return the point with smallest |f|
    best_x = None
    best_f = float('inf')
    for x_val, f_val in points:
        if math.isfinite(f_val) and abs(f_val) < best_f:
            best_f = abs(f_val)
            best_x = x_val
    
    return best_x if best_x is not None else x
# EVOLVE-BLOCK-END
