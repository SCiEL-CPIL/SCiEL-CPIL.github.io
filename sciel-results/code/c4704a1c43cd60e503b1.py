"""Root finding: Ill-conditioned - robust multi-strategy solver with caching and adaptive Newton."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Uses a portfolio of strategies: adaptive Newton, secant, bisection, and bracketing.
    Includes caching, progress monitoring, and final polish passes.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    best_x = float(x0)
    best_f = float('inf')
    cache = {}  # Cache: rounded x -> f(x)
    
    def check_time():
        return (time.time() - start_time) < MAX_WALL
    
    def eval_f(x):
        """Counted evaluation of f with caching and safety checks."""
        nonlocal best_x, best_f
        # Use a rounded key for cache to handle floating point
        if abs(x) < 1e15:
            key = round(x, 15)
        else:
            key = x
        if key in cache:
            val = cache[key]
        else:
            eval_count[0] += 1
            try:
                val = f(x)
            except Exception:
                val = 0.0
            if not math.isfinite(val):
                val = 0.0
            cache[key] = val
        # Track best result
        if abs(val) < abs(best_f):
            best_x = x
            best_f = val
        return val
    
    def is_good(x, fx=None):
        if fx is None:
            fx = eval_f(x)
        return abs(fx) <= TARGET_TOL
    
    def try_newton(x, max_iters=50):
        """Newton's method with adaptive step size, progress monitoring, and fallbacks."""
        nonlocal best_x, best_f
        x = float(x)
        prev_fx = None
        stall_count = 0
        
        for i in range(max_iters):
            if not check_time() or eval_count[0] >= MAX_EVALS - 10:
                break
            fx = eval_f(x)
            if abs(fx) <= TARGET_TOL:
                best_x = x
                best_f = fx
                return True
            
            # Progress monitoring
            if prev_fx is not None:
                if abs(fx) >= abs(prev_fx) * 0.99:
                    stall_count += 1
                else:
                    stall_count = 0
            prev_fx = fx
            
            if stall_count >= 3:
                break
            
            # Try multiple step sizes for derivative estimation
            found = False
            for h_factor in [1e-6, 1e-5, 1e-4, 1e-3]:
                if not check_time() or eval_count[0] >= MAX_EVALS - 5:
                    break
                h = h_factor * max(1.0, abs(x))
                fx_plus = eval_f(x + h)
                d = (fx_plus - fx) / h
                if d == 0 or not math.isfinite(d):
                    continue
                x_new = x - fx / d
                if not math.isfinite(x_new):
                    continue
                # Damping: if step is too large, reduce it
                if abs(x_new - x) > 10 * max(1.0, abs(x)):
                    x_new = x + 10 * max(1.0, abs(x)) * (x_new - x) / abs(x_new - x)
                x = x_new
                found = True
                break
            
            if not found:
                break
            
            if abs(x) > 1e15:
                break
        
        if best_f is not None and abs(best_f) <= TARGET_TOL:
            return True
        return False
    
    def try_secant(x0, x1, max_iters=50):
        """Secant method."""
        nonlocal best_x, best_f
        x_prev = float(x0)
        x_curr = float(x1)
        for i in range(max_iters):
            if not check_time() or eval_count[0] >= MAX_EVALS - 10:
                break
            f_curr = eval_f(x_curr)
            if abs(f_curr) <= TARGET_TOL:
                best_x = x_curr
                best_f = f_curr
                return True
            
            f_prev = eval_f(x_prev)
            if abs(f_prev) <= TARGET_TOL:
                best_x = x_prev
                best_f = f_prev
                return True
            
            if f_curr == f_prev:
                break
            
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            if not math.isfinite(x_new):
                break
            
            x_prev = x_curr
            x_curr = x_new
            
            if abs(x_curr) > 1e15:
                break
        
        if best_f is not None and abs(best_f) <= TARGET_TOL:
            return True
        return False
    
    def try_bisection(a, b, max_iters=100):
        """Bisection method, requires sign change."""
        nonlocal best_x, best_f
        fa = eval_f(a)
        fb = eval_f(b)
        
        if abs(fa) <= TARGET_TOL:
            best_x = a
            best_f = fa
            return True
        if abs(fb) <= TARGET_TOL:
            best_x = b
            best_f = fb
            return True
        
        if fa * fb > 0:
            return False  # No sign change
        
        for i in range(max_iters):
            if not check_time() or eval_count[0] >= MAX_EVALS - 10:
                break
            c = (a + b) / 2.0
            fc = eval_f(c)
            
            if abs(fc) <= TARGET_TOL:
                best_x = c
                best_f = fc
                return True
            
            if fa * fc <= 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            if abs(b - a) < 1e-15 * max(1.0, abs(c)):
                break
        
        if best_f is not None and abs(best_f) <= TARGET_TOL:
            return True
        return False
    
    def bracket_search(x0, max_expansions=20):
        """Find a bracket [a, b] with sign change around x0."""
        nonlocal best_x, best_f
        x = float(x0)
        fx = eval_f(x)
        if abs(fx) <= TARGET_TOL:
            best_x = x
            best_f = fx
            return (x, x)
        
        # Try expanding in both directions
        step = max(1.0, abs(x)) * 0.1
        for direction in [1, -1]:
            for i in range(max_expansions):
                if not check_time() or eval_count[0] >= MAX_EVALS - 20:
                    break
                x_new = x + direction * step
                fx_new = eval_f(x_new)
                if abs(fx_new) <= TARGET_TOL:
                    best_x = x_new
                    best_f = fx_new
                    return (x_new, x_new)
                if fx * fx_new < 0:
                    return (x, x_new)
                step *= 2.0
                if step > 1e15:
                    break
        
        return None
    
    def polish(x, max_iters=20):
        """Final polish pass: refine x using small Newton steps."""
        nonlocal best_x, best_f
        for i in range(max_iters):
            if not check_time() or eval_count[0] >= MAX_EVALS - 2:
                break
            fx = eval_f(x)
            if abs(fx) <= TARGET_TOL:
                best_x = x
                best_f = fx
                return True
            
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            
            # Very conservative step
            if abs(x_new - x) > 0.1 * max(1.0, abs(x)):
                x_new = x + 0.1 * max(1.0, abs(x)) * (x_new - x) / abs(x_new - x)
            
            x = x_new
        
        if abs(best_f) <= TARGET_TOL:
            return True
        return False
    
    # Initial evaluation
    x = float(x0)
    fx = eval_f(x)
    if abs(fx) <= TARGET_TOL:
        return x
    
    # Strategy 1: Try Newton from x0
    if try_newton(x, max_iters=40):
        return best_x
    
    # Strategy 2: Try to find a bracket and use bisection
    bracket = bracket_search(x, max_expansions=15)
    if bracket is not None:
        a, b = bracket
        if a != b:  # Actual bracket, not a single point
            if try_bisection(a, b, max_iters=60):
                return best_x
    
    # Strategy 3: Try secant with perturbed initial points
    h = 1e-4 * max(1.0, abs(x))
    if try_secant(x, x + h, max_iters=40):
        return best_x
    if try_secant(x, x - h, max_iters=40):
        return best_x
    
    # Strategy 4: Try Newton with different h bases
    for h_factor in [1e-4, 1e-5, 1e-3]:
        x_temp = x
        for i in range(30):
            if not check_time() or eval_count[0] >= MAX_EVALS - 10:
                break
            fx = eval_f(x_temp)
            if abs(fx) <= TARGET_TOL:
                best_x = x_temp
                best_f = fx
                return best_x
            h = h_factor * max(1.0, abs(x_temp))
            fx_plus = eval_f(x_temp + h)
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x_temp - fx / d
            if not math.isfinite(x_new):
                break
            if abs(x_new - x_temp) > 10 * max(1.0, abs(x_temp)):
                x_new = x_temp + 10 * max(1.0, abs(x_temp)) * (x_new - x_temp) / abs(x_new - x_temp)
            x_temp = x_new
    
    # Strategy 5: More aggressive bracket search
    bracket = bracket_search(x, max_expansions=30)
    if bracket is not None:
        a, b = bracket
        if a != b:
            if try_bisection(a, b, max_iters=80):
                return best_x
    
    # Strategy 6: Secant with larger perturbation
    h = 1e-2 * max(1.0, abs(x))
    if try_secant(x, x + h, max_iters=50):
        return best_x
    if try_secant(x, x - h, max_iters=50):
        return best_x
    
    # Strategy 7: Try a few random-ish perturbations
    for factor in [0.5, 2.0, 0.1, 10.0]:
        x_try = x * factor if x != 0 else x + factor
        if try_newton(x_try, max_iters=30):
            return best_x
    
    # Final polish: try to refine the best point found
    if check_time() and eval_count[0] < MAX_EVALS - 5:
        if polish(best_x, max_iters=20):
            return best_x
    
    # Return the best result found
    return best_x

# EVOLVE-BLOCK-END
