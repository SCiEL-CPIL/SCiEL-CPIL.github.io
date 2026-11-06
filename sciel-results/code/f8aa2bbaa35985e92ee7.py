"""Root finding: Ill-conditioned - robust multi-strategy solver with caching and Brent's method."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Uses caching, Newton, Brent's method, and secant with adaptive strategies.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    best_x = float(x0)
    best_f = None
    cache = {}  # Cache: x -> f(x)
    
    def check_time():
        return (time.time() - start_time) < MAX_WALL
    
    def eval_f(x):
        """Counted evaluation of f with caching and safety checks."""
        nonlocal best_x, best_f
        # Use a rounded key for cache to handle floating point
        key = round(x, 15) if abs(x) < 1e15 else x
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
        if best_f is None or abs(val) < abs(best_f):
            best_x = x
            best_f = val
        return val
    
    def try_newton(x, max_iters=50, h_factor=1e-6):
        """Newton's method with single h and damping."""
        nonlocal best_x, best_f
        x = float(x)
        for i in range(max_iters):
            if not check_time() or eval_count[0] >= MAX_EVALS - 5:
                break
            fx = eval_f(x)
            if abs(fx) <= TARGET_TOL:
                best_x = x
                best_f = fx
                return True
            
            h = h_factor * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                # Try larger h
                h = 1e-4 * max(1.0, abs(x))
                fx_plus = eval_f(x + h)
                d = (fx_plus - fx) / h
                if d == 0 or not math.isfinite(d):
                    break
            
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            # Damping
            if abs(x_new - x) > 5 * max(1.0, abs(x)):
                x_new = x + 5 * max(1.0, abs(x)) * (x_new - x) / abs(x_new - x)
            x = x_new
            if abs(x) > 1e15:
                break
        
        if best_f is not None and abs(best_f) <= TARGET_TOL:
            return True
        return False
    
    def try_brent(a, b, fa, fb, max_iters=100):
        """
        Brent's method: combines bisection, secant, and inverse quadratic interpolation.
        Requires fa and fb with opposite signs.
        """
        nonlocal best_x, best_f
        a = float(a)
        b = float(b)
        fa = float(fa)
        fb = float(fb)
        
        if abs(fa) <= TARGET_TOL:
            best_x = a
            best_f = fa
            return True
        if abs(fb) <= TARGET_TOL:
            best_x = b
            best_f = fb
            return True
        
        if fa * fb > 0:
            return False
        
        c = a
        fc = fa
        d = b - a
        e = d
        
        for i in range(max_iters):
            if not check_time() or eval_count[0] >= MAX_EVALS - 5:
                break
            
            # Check if a or b is good
            if abs(fa) <= TARGET_TOL:
                best_x = a
                best_f = fa
                return True
            if abs(fb) <= TARGET_TOL:
                best_x = b
                best_f = fb
                return True
            
            # Inverse quadratic interpolation
            if fa != fc and fb != fc:
                s = b * fc * fa / ((fb - fc) * (fb - fa)) + c * fa * fb / ((fc - fa) * (fc - fb)) + a * fb * fc / ((fb - fa) * (fb - fc))
            else:
                # Secant
                s = b - fb * (b - a) / (fb - fa)
            
            # Check if s is acceptable
            cond1 = (s < min(a, b) or s > max(a, b))
            cond2 = (abs(s - b) >= abs(b - c) / 2)
            cond3 = (abs(s - c) < abs(c - d) / 2)
            cond4 = (abs(e) < abs(c - d) / 2)
            
            if cond1 or cond2:
                # Use bisection
                s = (a + b) / 2
                e = d
            else:
                d = e
                e = b - s
            
            c = b
            fc = fb
            d = b - s
            
            if abs(s - b) < 1e-15 * max(1.0, abs(s)):
                # Convergence
                break
            
            b = s
            fb = eval_f(b)
            
            if fa * fb < 0:
                a = c
                fa = fc
            else:
                a = b
                fa = fb
            
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            
            if abs(b - a) < 1e-15 * max(1.0, abs(b)):
                break
        
        # Return the better of a and b
        if abs(fa) <= abs(fb):
            if abs(fa) <= TARGET_TOL:
                best_x = a
                best_f = fa
                return True
        else:
            if abs(fb) <= TARGET_TOL:
                best_x = b
                best_f = fb
                return True
        
        return best_f is not None and abs(best_f) <= TARGET_TOL
    
    def try_secant(x0, x1, max_iters=50):
        """Secant method."""
        nonlocal best_x, best_f
        x_prev = float(x0)
        x_curr = float(x1)
        for i in range(max_iters):
            if not check_time() or eval_count[0] >= MAX_EVALS - 5:
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
    
    def bracket_search(x0, max_expansions=40):
        """Find a bracket [a, b] with sign change around x0."""
        nonlocal best_x, best_f
        x = float(x0)
        fx = eval_f(x)
        if abs(fx) <= TARGET_TOL:
            best_x = x
            best_f = fx
            return (x, x, fx)
        
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
                    return (x_new, x_new, fx_new)
                if fx * fx_new < 0:
                    return (x, x_new, fx)
                step *= 2.0
                if step > 1e15:
                    break
        
        return None
    
    # Initial evaluation
    x = float(x0)
    fx = eval_f(x)
    if abs(fx) <= TARGET_TOL:
        return x
    
    # Strategy 1: Try Newton from x0 with small h
    if try_newton(x, max_iters=30, h_factor=1e-6):
        return best_x
    
    # Strategy 2: Try to find a bracket and use Brent's method
    bracket = bracket_search(x, max_expansions=30)
    if bracket is not None:
        a, b, fa = bracket
        if a != b:  # Actual bracket
            fb = eval_f(b)
            if try_brent(a, b, fa, fb, max_iters=80):
                return best_x
    
    # Strategy 3: Try secant with perturbed initial points
    h = 1e-5 * max(1.0, abs(x))
    if try_secant(x, x + h, max_iters=30):
        return best_x
    if try_secant(x, x - h, max_iters=30):
        return best_x
    
    # Strategy 4: Try Newton with larger h
    if try_newton(x, max_iters=30, h_factor=1e-4):
        return best_x
    
    # Strategy 5: More aggressive bracket search
    bracket = bracket_search(x, max_expansions=50)
    if bracket is not None:
        a, b, fa = bracket
        if a != b:
            fb = eval_f(b)
            if try_brent(a, b, fa, fb, max_iters=100):
                return best_x
    
    # Strategy 6: Secant with larger perturbation
    h = 1e-3 * max(1.0, abs(x))
    if try_secant(x, x + h, max_iters=40):
        return best_x
    if try_secant(x, x - h, max_iters=40):
        return best_x
    
    # Strategy 7: Try Newton from perturbed points
    for factor in [0.5, 2.0, 0.1, 10.0]:
        x_try = x * factor if x != 0 else x + factor
        if try_newton(x_try, max_iters=25, h_factor=1e-5):
            return best_x
    
    # Strategy 8: Final secant attempts with various h
    for h_factor in [1e-2, 1e-1]:
        h = h_factor * max(1.0, abs(x))
        if try_secant(x, x + h, max_iters=30):
            return best_x
        if try_secant(x, x - h, max_iters=30):
            return best_x
    
    # Strategy 9: Final polish - try high-precision Newton from best_x
    if best_f is not None and abs(best_f) > TARGET_TOL and abs(best_f) < 1e-6:
        # If we're close but not quite there, try a more precise Newton
        if try_newton(best_x, max_iters=20, h_factor=1e-8):
            return best_x
        # Try secant with very small perturbation
        h = 1e-8 * max(1.0, abs(best_x))
        if try_secant(best_x, best_x + h, max_iters=20):
            return best_x
        if try_secant(best_x, best_x - h, max_iters=20):
            return best_x
    
    # Return the best result found
    return best_x

# EVOLVE-BLOCK-END
