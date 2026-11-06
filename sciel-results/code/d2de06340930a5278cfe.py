"""Root finding: Ill-conditioned - Illinois method with adaptive bracketing and secant polish."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Strategy:
    1. Quick check at x0 and immediate neighbors
    2. Adaptive step-doubling to find a sign-change bracket
    3. Illinois method (modified bisection/secant) for guaranteed convergence
    4. Secant polish from the bracket midpoint for high precision
    5. Final Newton-like refinement with central differences
    """
    start_time = time.time()
    MAX_WALL = 118.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]

    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        if time.time() - start_time > MAX_WALL:
            raise RuntimeError("Time limit exceeded")
        
        eval_count[0] += 1
        try:
            val = f(x)
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        return val

    def is_good(x, fx, tol=TARGET_TOL):
        return abs(fx) <= tol

    def check_budget():
        return eval_count[0] < MAX_EVALS and (time.time() - start_time < MAX_WALL)

    def elapsed():
        return time.time() - start_time

    # --- Strategy 1: Adaptive step-doubling bracket finder ---
    def find_bracket(x0):
        """Find [a, b] such that f(a)*f(b) < 0 using adaptive step doubling."""
        if not check_budget():
            return None
        
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return (x, x, fx, fx)
        
        # Start with a small step
        h = 1e-4 * max(1.0, abs(x))
        
        # Try expanding in both directions
        for direction in [1, -1]:
            if not check_budget():
                break
            
            prev_x = x
            prev_fx = fx
            cur_h = h
            
            for i in range(50):
                if not check_budget():
                    break
                
                cur_x = x + direction * cur_h
                cur_fx = eval_f(cur_x)
                
                if is_good(cur_x, cur_fx):
                    return (cur_x, cur_x, cur_fx, cur_fx)
                
                if prev_fx * cur_fx < 0:
                    return (prev_x, cur_x, prev_fx, cur_fx)
                
                # Double the step size
                cur_h *= 2.0
                prev_x = cur_x
                prev_fx = cur_fx
                
                # Stop if we're going too far
                if abs(cur_h) > 1e6 * max(1.0, abs(x)):
                    break
        
        # If no sign change found, try a wider initial scan
        for i in range(1, 20):
            if not check_budget():
                break
            for direction in [1, -1]:
                if not check_budget():
                    break
                cur_x = x + direction * (1e-3 * max(1.0, abs(x)) * (10 ** i))
                cur_fx = eval_f(cur_x)
                if is_good(cur_x, cur_fx):
                    return (cur_x, cur_x, cur_fx, cur_fx)
                if fx * cur_fx < 0:
                    return (x, cur_x, fx, cur_fx)
        
        return None

    # --- Strategy 2: Illinois Method ---
    def illinois_method(a, b, fa, fb, max_iter=100):
        """
        Illinois method: modified bisection that uses secant on the larger
        sub-interval and shrinks it by a fixed factor (0.5) instead of bisection.
        This is specifically designed for ill-conditioned functions.
        """
        if not check_budget():
            return None
        
        if fa * fb > 0:
            return None
        
        # Ensure |fa| < |fb| for the Illinois method
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        for i in range(max_iter):
            if not check_budget():
                break
            
            # If we're close enough, return
            if is_good(a, fa):
                return a
            if is_good(b, fb):
                return b
            
            # Secant point
            if fa == fb:
                # Degenerate case, use bisection
                c = 0.5 * (a + b)
            else:
                c = (a * fb - b * fa) / (fb - fa)
            
            if not math.isfinite(c) or c <= a or c >= b:
                c = 0.5 * (a + b)
            
            fc = eval_f(c)
            if is_good(c, fc):
                return c
            
            # Update bracket
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
            
            # Illinois modification: shrink the larger interval by 0.5
            if abs(fa) > abs(fb):
                # Shrink a-side: a = a + 0.5*(b-a)
                new_a = a + 0.5 * (b - a)
                if new_a != a:
                    fa = eval_f(new_a)
                    a = new_a
                    if is_good(a, fa):
                        return a
            else:
                # Shrink b-side: b = b + 0.5*(a-b)
                new_b = b + 0.5 * (a - b)
                if new_b != b:
                    fb = eval_f(new_b)
                    b = new_b
                    if is_good(b, fb):
                        return b
            
            # Check convergence
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        
        # Return the point with smaller |f|
        return a if abs(fa) <= abs(fb) else b

    # --- Strategy 3: Secant polish ---
    def secant_polish(x0, max_iter=30):
        """High-precision secant method for final refinement."""
        if not check_budget():
            return None
        
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        
        h = 1e-5 * max(1.0, abs(x))
        x_prev = x + h
        fx_prev = eval_f(x_prev)
        if is_good(x_prev, fx_prev):
            return x_prev
        
        for i in range(max_iter):
            if not check_budget():
                break
            
            if fx == fx_prev:
                break
            if abs(x - x_prev) < 1e-14 * max(1.0, abs(x)):
                break
            
            d = (fx - fx_prev) / (x - x_prev)
            if d == 0 or not math.isfinite(d):
                break
            
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            
            # Only accept if we made progress
            if abs(fx_new) < abs(fx):
                x_prev, fx_prev = x, fx
                x, fx = x_new, fx_new
            else:
                # Damp the step
                x_new = x - 0.5 * (fx / d)
                if not math.isfinite(x_new):
                    break
                fx_new = eval_f(x_new)
                if is_good(x_new, fx_new):
                    return x_new
                if abs(fx_new) < abs(fx):
                    x_prev, fx_prev = x, fx
                    x, fx = x_new, fx_new
                else:
                    break
            
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
                break
        
        return x if abs(fx) <= TARGET_TOL else None

    # --- Strategy 4: Central-difference Newton polish ---
    def newton_polish(x0, max_iter=20):
        """Newton method with central differences for final precision."""
        if not check_budget():
            return None
        
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        
        for i in range(max_iter):
            if not check_budget():
                break
            if abs(fx) <= TARGET_TOL:
                return x
            
            h = 1e-5 * max(1.0, abs(x))
            if h < 1e-13:
                h = 1e-13
            
            fx_p = eval_f(x + h)
            if is_good(x + h, fx_p):
                return x + h
            fx_m = eval_f(x - h)
            if is_good(x - h, fx_m):
                return x - h
            
            d = (fx_p - fx_m) / (2 * h)
            
            if d == 0 or not math.isfinite(d):
                break
            
            dx = fx / d
            if not math.isfinite(dx):
                break
            
            x_new = x - dx
            if not math.isfinite(x_new):
                break
            
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Damped step
                x_d = x - 0.5 * dx
                if math.isfinite(x_d):
                    fx_d = eval_f(x_d)
                    if is_good(x_d, fx_d):
                        return x_d
                    if abs(fx_d) < abs(fx):
                        x, fx = x_d, fx_d
                    else:
                        break
                else:
                    break
            
            if abs(dx) < 1e-14 * max(1.0, abs(x)):
                break
        
        return x if abs(fx) <= TARGET_TOL else None

    # === MAIN EXECUTION ===
    
    # 1. Quick check at x0
    fx0 = eval_f(float(x0))
    if is_good(x0, fx0):
        return float(x0)
    
    # 2. Find a bracket
    bracket = find_bracket(x0)
    
    if bracket is not None:
        a, b, fa, fb = bracket
        
        if abs(a - b) < 1e-15:
            # Degenerate bracket, just return the good point
            return a if is_good(a, fa) else b
        
        # 3. Apply Illinois method
        result = illinois_method(a, b, fa, fb)
        if result is not None:
            # 4. Polish with secant
            polished = secant_polish(result)
            if polished is not None:
                return polished
            # 5. Polish with Newton
            polished = newton_polish(result)
            if polished is not None:
                return polished
            return result
    
    # 6. If no bracket found, try secant from x0
    result = secant_polish(x0)
    if result is not None:
        return result
    
    # 7. Try Newton from best point found
    result = newton_polish(best_x[0])
    if result is not None:
        return result
    
    # 8. Try secant from best point if different from x0
    if abs(best_x[0] - float(x0)) > 1e-6:
        result = secant_polish(best_x[0])
        if result is not None:
            return result
    
    # 9. Final fallback: return best found
    return best_x[0]

# EVOLVE-BLOCK-END
