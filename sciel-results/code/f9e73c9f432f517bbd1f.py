"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 140  # 140s safety margin (evaluator kills at 150s)
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    
    def safe_f(x):
        nonlocal eval_count
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded")
        if eval_count >= max_evals:
            raise ValueError("Max evaluations exceeded")
        eval_count += 1
        val = f(x)
        if not math.isfinite(val):
            # If f returns inf or nan, return a large number to push the bracket away
            # or handle specifically. For now, return the value, let logic handle it.
            return val
        return float(val)

    x = float(x0)
    
    # Phase 1: Find a bracket [a, b] such that f(a) and f(b) have opposite signs.
    # Start with x0. If f(x0) is small, return.
    # Expand interval around x0.
    
    a = x
    fa = safe_f(a)
    if abs(fa) <= tol:
        return a

    b = x
    fb = None
    found_bracket = False
    
    # Try to find a sign change by expanding
    # We keep track of the best point (smallest |f|)
    best_x = a
    best_f = fa
    best_abs_f = abs(fa)
    
    # Initial expansion step
    # Use a heuristic step size
    step = 1.0 if abs(x) < 1.0 else abs(x) * 0.1
    
    # We will try points in sequence: x0, x0+step, x0-step, x0+2*step, etc.
    # But we need a bracket.
    
    # Let's use a simpler robust bracketing:
    # 1. Check x0.
    # 2. Check x0 +/- delta.
    # 3. Expand.
    
    # Reset for bracketing search
    a = x
    fa = best_f
    b = None
    fb = None
    
    # If we already have a candidate, let's try to bracket around it.
    # We need two points with opposite signs.
    
    # Try immediate neighbors
    candidates = [x, x + step, x - step]
    
    points = []
    for cx in candidates:
        cf = safe_f(cx)
        if abs(cf) < best_abs_f:
            best_abs_f = abs(cf)
            best_x = cx
            best_f = cf
        points.append((cx, cf))
        
        if abs(cf) <= tol:
            return cx
            
        # Check for sign change with previous points
        for px, pf in points[:-1]:
            if pf * cf < 0:
                a, b = px, cx
                fa, fb = pf, cf
                found_bracket = True
                break
        if found_bracket:
            break
            
    if not found_bracket:
        # Expand bracket
        # Pick the point with smallest |f| as center? Or just expand from x0?
        # Let's expand from the point with smallest |f|
        center = best_x
        f_center = best_f
        
        # We need a point with opposite sign.
        # Try expanding in both directions from center
        left_x = center - step
        right_x = center + step
        
        # We already evaluated some points. Let's just do a systematic expansion.
        # To save evals, we reuse known points if possible, but it's easier to just evaluate new ones.
        
        # Let's try a sequence of points: center +/- k*step
        # We want to find a sign change with f_center.
        
        found_bracket = False
        for k in range(1, 100):
            if time.time() > deadline:
                break
                
            # Try right
            rx = center + k * step
            fr = safe_f(rx)
            if abs(fr) < best_abs_f:
                best_abs_f = abs(fr)
                best_x = rx
                best_f = fr
            if abs(fr) <= tol:
                return rx
            if f_center * fr < 0:
                a, b = center, rx
                fa, fb = f_center, fr
                found_bracket = True
                break
                
            # Try left
            lx = center - k * step
            fl = safe_f(lx)
            if abs(fl) < best_abs_f:
                best_abs_f = abs(fl)
                best_x = lx
                best_f = fl
            if abs(fl) <= tol:
                return lx
            if f_center * fl < 0:
                a, b = lx, center
                fa, fb = fl, f_center
                found_bracket = True
                break
                
            # Increase step size exponentially to cover large ranges
            if k % 5 == 0:
                step *= 2.0

    if not found_bracket:
        # If no bracket found, return the best point we have
        return best_x

    # Phase 2: Root finding within [a, b]
    # We have a, b with fa, fb. fa*fb < 0.
    # We also have a third point 'c' which is the best guess so far.
    # Initially, let c be the midpoint or the point with smaller |f|?
    # Standard Brent's method uses the previous point. 
    # Let's set c to the point among a, b with smaller |f|? 
    # Or just start with c = (a+b)/2?
    # Actually, for IQI we need 3 distinct points.
    # Let's set c to the best_x if it's in [a,b], else midpoint.
    
    if a <= best_x <= b:
        c = best_x
        fc = best_f
    else:
        c = (a + b) / 2.0
        fc = safe_f(c)
        if abs(fc) <= tol:
            return c
        # Update bracket if c has smaller |f|? No, bracket is fixed by sign change.
        # But we should ensure c is distinct from a and b.
        if c == a or c == b:
             c = (a + b) / 2.0
             fc = safe_f(c)

    # Ensure a, b, c are distinct
    if a == b or a == c or b == c:
         # Fallback to bisection if points are not distinct
         for _ in range(100):
             if time.time() > deadline: break
             m = (a + b) / 2.0
             fm = safe_f(m)
             if abs(fm) <= tol:
                 return m
             if fa * fm < 0:
                 b, fb = m, fm
             else:
                 a, fa = m, fm
             if abs(b - a) < 1e-15:
                 break
         return (a + b) / 2.0

    # Main IQI Loop
    for _ in range(200): # Max iterations, usually converges much faster
        if time.time() > deadline:
            break
            
        # Check termination
        if abs(fa) <= tol: return a
        if abs(fb) <= tol: return b
        if abs(fc) <= tol: return c
        
        # Check if interval is small enough
        if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
            # Return the point with smallest |f|
            if abs(fa) <= abs(fb) and abs(fa) <= abs(fc): return a
            if abs(fb) <= abs(fa) and abs(fb) <= abs(fc): return b
            return c

        # Calculate IQI step
        # Points: (a, fa), (b, fb), (c, fc)
        # We want x such that y=0 for x(y)
        
        # To avoid numerical issues, we can use the formula derived from Lagrange basis.
        # x = x1 * y2*y3 / ((y1-y2)*(y1-y3)) + ...
        
        y1, y2, y3 = fa, fb, fc
        x1, x2, x3 = a, b, c
        
        # Check for degenerate cases (duplicate y values)
        if y1 == y2 or y1 == y3 or y2 == y3:
            # Fallback to Bisection
            x_new = (a + b) / 2.0
        else:
            # Compute terms
            # term1 = x1 * y2 * y3 / ((y1 - y2) * (y1 - y3))
            # term2 = x2 * y1 * y3 / ((y2 - y1) * (y2 - y3))
            # term3 = x3 * y1 * y2 / ((y3 - y1) * (y3 - y2))
            
            try:
                t1 = x1 * (y2 * y3) / ((y1 - y2) * (y1 - y3))
                t2 = x2 * (y1 * y3) / ((y2 - y1) * (y2 - y3))
                t3 = x3 * (y1 * y2) / ((y3 - y1) * (y3 - y2))
                x_new = t1 + t2 + t3
            except ZeroDivisionError:
                x_new = (a + b) / 2.0
                
            # Validate IQI step
            # It must be within the bracket [min(a,b), max(a,b)]
            lo = min(a, b)
            hi = max(a, b)
            
            # Also, it shouldn't be too close to the endpoints to avoid instability
            # A common heuristic is that it should be within 1/4 of the interval width from the center?
            # Or just strictly inside.
            if lo < x_new < hi:
                # Check if it's not too close to a or b
                # If |x_new - a| < 1e-15 * |a| or similar, it might be numerically unstable
                # Let's just accept it if it's inside.
                pass
            else:
                # Fallback to Bisection
                x_new = (a + b) / 2.0
                
        # Evaluate f at x_new
        f_new = safe_f(x_new)
        
        # Update best
        if abs(f_new) < best_abs_f:
            best_abs_f = abs(f_new)
            best_x = x_new
            best_f = f_new
            
        if abs(f_new) <= tol:
            return x_new
            
        # Update Bracket
        # We need to maintain fa*fb < 0
        # If f(a) and f_new have opposite signs, new bracket is [a, x_new]
        # If f(b) and f_new have opposite signs, new bracket is [x_new, b]
        
        if fa * f_new < 0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new
            
        # Update c
        # c should be the point among a, b, x_new (old c is replaced or kept?)
        # Standard Brent's method keeps the previous point.
        # For IQI, we want the 3 most recent/best points.
        # Let's set c to the point with the second smallest |f| among a, b, and the old c?
        # Or simpler: c is the point that is NOT a or b in the new set {a, b, x_new}?
        # Wait, x_new is either a or b now.
        # So the set of points is {a, b, old_c}.
        # We should discard the point with the largest |f| to keep the best 3?
        # Or just keep old_c if it's still in the bracket?
        
        # Let's follow a simple strategy:
        # The 3 points for the next iteration are:
        # 1. a
        # 2. b
        # 3. The point among {old_a, old_b, old_c, x_new} that is NOT a or b, and has smallest |f|?
        # Actually, x_new is one of a or b.
        # So we have 3 candidates: new_a, new_b, and old_c.
        # If old_c is outside [new_a, new_b], we should probably replace it with the midpoint or the other endpoint?
        # No, IQI works best if points are distinct and close.
        
        # Let's just keep old_c as c, unless it's identical to new a or b.
        # If old_c == new_a or old_c == new_b, we need a new c.
        # Let's set c to the midpoint of a and b if it's not already a or b.
        
        if c == a or c == b:
             c = (a + b) / 2.0
             fc = safe_f(c)
             if abs(fc) <= tol:
                 return c
        else:
             # c is distinct from a and b.
             # But is it in the bracket?
             if not (min(a, b) < c < max(a, b)):
                  # c is outside. Let's move it to midpoint.
                  c = (a + b) / 2.0
                  fc = safe_f(c)
                  if abs(fc) <= tol:
                      return c

    # Return best found
    return best_x

# EVOLVE-BLOCK-END
