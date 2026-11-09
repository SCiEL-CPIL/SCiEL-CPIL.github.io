"""Root finding for ill-conditioned equations: Robust Adaptive Hybrid."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 450   # Safety margin below 500 limit
    
    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    
    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS:
            return True
        if time.time() - start_time > DEADLINE:
            return True
        return False
    
    def safe_eval(x):
        nonlocal eval_count, best_x, best_f
        if check_budget():
            return None
        try:
            val = f(x)
            eval_count += 1
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                return None
            val = float(val)
            if best_f is None or abs(val) < abs(best_f):
                best_f = val
                best_x = x
            return val
        except Exception:
            return None
    
    def try_bracket(x_center, f_center):
        """Try to find a bracket around x_center."""
        nonlocal eval_count, best_x, best_f
        
        if f_center is None or abs(f_center) <= 1e-8:
            return None
        
        # Try expanding bracket
        step = max(1e-4, abs(x_center) * 1e-4)
        for attempt in range(20):
            if check_budget():
                return None
            
            x_left = x_center - step
            x_right = x_center + step
            
            f_left = safe_eval(x_left)
            if f_left is None:
                step *= 2
                continue
            if abs(f_left) <= 1e-8:
                return x_left
            
            f_right = safe_eval(x_right)
            if f_right is None:
                step *= 2
                continue
            if abs(f_right) <= 1e-8:
                return x_right
            
            if f_left * f_right < 0:
                return (x_left, x_right)
            
            step *= 2
        
        return None
    
    def bisection(a, b, fa, fb, max_iter=50):
        """Bisection method."""
        nonlocal eval_count, best_x, best_f
        
        if fa * fb > 0:
            return None
        
        for _ in range(max_iter):
            if check_budget():
                break
            
            mid = (a + b) / 2.0
            f_mid = safe_eval(mid)
            if f_mid is None:
                break
            
            if abs(f_mid) <= 1e-8:
                return mid
            
            if fa * f_mid <= 0:
                b = mid
                fb = f_mid
            else:
                a = mid
                fa = f_mid
        
        # Return the best of the endpoints and midpoint
        candidates = [(a, fa), (b, fb)]
        if mid is not None and f_mid is not None:
            candidates.append((mid, f_mid))
        
        best = min(candidates, key=lambda t: abs(t[1]))
        return best[0]
    
    def secant(x0, x1, f0, f1, max_iter=50):
        """Secant method."""
        nonlocal eval_count, best_x, best_f
        
        x_prev = x0
        x_curr = x1
        f_prev = f0
        f_curr = f1
        
        for _ in range(max_iter):
            if check_budget():
                break
            
            if abs(f_curr) <= 1e-8:
                return x_curr
            
            if f_curr == f_prev:
                break
            
            dx = (f_curr - f_prev) / (x_curr - x_prev) if x_curr != x_prev else 0
            if dx == 0:
                break
            
            x_next = x_curr - f_curr / dx
            
            f_next = safe_eval(x_next)
            if f_next is None:
                break
            
            x_prev = x_curr
            x_curr = x_next
            f_prev = f_curr
            f_curr = f_next
        
        return x_curr
    
    def newton(x, f_val, max_iter=30):
        """Newton's method with finite difference derivative."""
        nonlocal eval_count, best_x, best_f
        
        x_curr = x
        f_curr = f_val
        
        for _ in range(max_iter):
            if check_budget():
                break
            
            if abs(f_curr) <= 1e-8:
                return x_curr
            
            h = 1e-8 * max(1.0, abs(x_curr))
            f_plus = safe_eval(x_curr + h)
            if f_plus is None:
                break
            
            d = (f_plus - f_curr) / h
            if d == 0 or math.isnan(d) or math.isinf(d):
                break
            
            x_next = x_curr - f_curr / d
            
            # Safeguard: if Newton step is too large, damp it
            if abs(x_next - x_curr) > 10 * max(1.0, abs(x_curr)):
                x_next = x_curr + 0.5 * (x_next - x_curr)
            
            f_next = safe_eval(x_next)
            if f_next is None:
                break
            
            # If Newton made things worse, try a smaller step
            if abs(f_next) > 2 * abs(f_curr):
                x_next = x_curr - 0.5 * (x_curr - x_next)
                f_next = safe_eval(x_next)
                if f_next is None:
                    break
            
            x_curr = x_next
            f_curr = f_next
        
        return x_curr
    
    # Main algorithm
    x = float(x0)
    fx = safe_eval(x)
    
    if fx is None:
        return best_x
    
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 1: Try Newton's method
    x_newton = newton(x, fx, max_iter=20)
    if x_newton is not None and abs(safe_eval(x_newton) if safe_eval(x_newton) is not None else 1) <= 1e-8:
        return x_newton
    
    # Update best if Newton improved things
    if x_newton is not None:
        fx_new = safe_eval(x_newton)
        if fx_new is not None and (best_f is None or abs(fx_new) < abs(best_f)):
            best_x = x_newton
            best_f = fx_new
    
    # Phase 2: Try to find a bracket and use bisection
    bracket = try_bracket(x, fx)
    if bracket is not None:
        if isinstance(bracket, tuple):
            a, b = bracket
            fa = safe_eval(a)
            fb = safe_eval(b)
            if fa is not None and fb is not None and fa * fb < 0:
                x_bisect = bisection(a, b, fa, fb, max_iter=60)
                if x_bisect is not None:
                    fx_bisect = safe_eval(x_bisect)
                    if fx_bisect is not None and abs(fx_bisect) <= 1e-8:
                        return x_bisect
                    if fx_bisect is not None and (best_f is None or abs(fx_bisect) < abs(best_f)):
                        best_x = x_bisect
                        best_f = fx_bisect
        else:
            # Found a root directly
            if abs(safe_eval(bracket) if safe_eval(bracket) is not None else 1) <= 1e-8:
                return bracket
    
    # Phase 3: Try secant method from best points
    if best_x is not None and best_f is not None:
        # Use x and best_x as secant points if they're different
        if abs(x - best_x) > 1e-12:
            fx_best = safe_eval(best_x)
            if fx_best is not None:
                x_secant = secant(x, best_x, fx, fx_best, max_iter=40)
                if x_secant is not None:
                    fx_secant = safe_eval(x_secant)
                    if fx_secant is not None and abs(fx_secant) <= 1e-8:
                        return x_secant
                    if fx_secant is not None and (best_f is None or abs(fx_secant) < abs(best_f)):
                        best_x = x_secant
                        best_f = fx_secant
    
    # Phase 4: Try multiple starting points with Newton
    offsets = [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]
    for offset in offsets:
        if check_budget():
            break
        
        x_try = x + offset
        fx_try = safe_eval(x_try)
        if fx_try is None:
            continue
        
        if abs(fx_try) <= 1e-8:
            return x_try
        
        x_n = newton(x_try, fx_try, max_iter=15)
        if x_n is not None:
            fx_n = safe_eval(x_n)
            if fx_n is not None and abs(fx_n) <= 1e-8:
                return x_n
            if fx_n is not None and (best_f is None or abs(fx_n) < abs(best_f)):
                best_x = x_n
                best_f = fx_n
    
    # Phase 5: Final polish with tighter tolerance if we have a good candidate
    if best_f is not None and abs(best_f) < 1e-6:
        x_polish = newton(best_x, best_f, max_iter=20)
        if x_polish is not None:
            fx_polish = safe_eval(x_polish)
            if fx_polish is not None and abs(fx_polish) <= 1e-8:
                return x_polish
            if fx_polish is not None and (best_f is None or abs(fx_polish) < abs(best_f)):
                best_x = x_polish
                best_f = fx_polish
    
    return best_x
# EVOLVE-BLOCK-END
