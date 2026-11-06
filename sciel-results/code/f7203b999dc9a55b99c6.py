"""Root finding: Ill-conditioned - Illinois method with adaptive bracketing and polish."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses Illinois method (modified bisection) with adaptive bracketing.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    
    def eval_f(x):
        """Counted evaluation of f."""
        eval_count[0] += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return 0.0
            return float(val)
        except Exception:
            return 0.0
    
    def check_time():
        """Check if we're close to the time limit."""
        return (time.time() - start_time) > MAX_WALL
    
    def is_good(x, fx):
        """Check if x is a good root."""
        return abs(fx) <= TARGET_TOL
    
    def illinois_method(a, fa, b, fb, max_iter=300):
        """
        Illinois method: modified bisection where the point with the larger
        |f| is moved to the midpoint, while the other endpoint stays fixed.
        This accelerates convergence for ill-conditioned functions.
        """
        # Ensure |fa| <= |fb| so we move b (the larger one)
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        for i in range(max_iter):
            if check_time() or eval_count[0] >= MAX_EVALS:
                break
            
            # Illinois update: move the endpoint with larger |f| to midpoint
            # Since |fa| <= |fb|, we move b to midpoint
            c = (a + b) / 2.0
            fc = eval_f(c)
            
            if not math.isfinite(fc):
                break
            
            if is_good(c, fc):
                return c
            
            # Update bracket
            if fa * fc <= 0:
                # Root is in [a, c]
                b = c
                fb = fc
            else:
                # Root is in [c, b]
                a = c
                fa = fc
            
            # Illinois: shrink the interval by moving the larger-|f| endpoint
            # to the midpoint, keeping the smaller-|f| endpoint fixed
            if abs(fa) > abs(fb):
                # fa is larger, move a to midpoint of current [a,b]
                a = (a + b) / 2.0
                fa = eval_f(a)
                if not math.isfinite(fa):
                    break
                if is_good(a, fa):
                    return a
            else:
                # fb is larger, move b to midpoint of current [a,b]
                b = (a + b) / 2.0
                fb = eval_f(b)
                if not math.isfinite(fb):
                    break
                if is_good(b, fb):
                    return b
            
            # Check for convergence
            if abs(b - a) < 1e-15 * max(1.0, abs(a)):
                break
        
        # Return the point with smaller |f|
        if abs(fa) < abs(fb):
            return a
        else:
            return b
    
    def find_bracket(x0, f_x0, max_attempts=50):
        """
        Find a sign-change bracket around x0 using adaptive step-doubling.
        Returns (a, fa, b, fb) where fa*fb < 0, or None.
        """
        if f_x0 == 0:
            return x0, f_x0, x0, f_x0
        
        scale = max(1e-10, abs(x0) * 1e-10)
        if scale == 0:
            scale = 1e-10
        
        # Try expanding in both directions
        for direction in [1, -1]:
            step = scale
            for i in range(max_attempts):
                if check_time() or eval_count[0] >= MAX_EVALS:
                    return None
                
                x_try = x0 + direction * step
                f_try = eval_f(x_try)
                
                if not math.isfinite(f_try):
                    step *= 2
                    continue
                
                if is_good(x_try, f_try):
                    return x_try, f_try, x_try, f_try
                
                if f_x0 * f_try < 0:
                    if f_x0 < f_try:
                        return x0, f_x0, x_try, f_try
                    else:
                        return x_try, f_try, x0, f_x0
                
                step *= 2.0
                if step > 1e15:
                    break
        
        return None
    
    def polish(x, fx, max_iter=50):
        """
        Final high-precision polish using Illinois method on a tiny bracket.
        """
        if is_good(x, fx):
            return x
        
        # Create a tiny bracket around x
        h = 1e-12 * max(1.0, abs(x))
        a = x - h
        b = x + h
        
        fa = eval_f(a)
        fb = eval_f(b)
        
        if not math.isfinite(fa) or not math.isfinite(fb):
            return x
        
        # Track best
        best_x = x
        best_fx = fx
        if abs(fa) < abs(best_fx):
            best_x, best_fx = a, fa
        if abs(fb) < abs(best_fx):
            best_x, best_fx = b, fb
        
        if is_good(best_x, best_fx):
            return best_x
        
        # If we have a sign change, use Illinois
        if fa * fb < 0:
            result = illinois_method(a, fa, b, fb, max_iter=30)
            if result is not None:
                fr = eval_f(result)
                if is_good(result, fr):
                    return result
                if abs(fr) < abs(best_fx):
                    best_x, best_fx = result, fr
        
        # Try secant-based polish
        x1 = x
        f1 = fx
        x2 = x + h
        f2 = eval_f(x2)
        
        for i in range(30):
            if check_time() or eval_count[0] >= MAX_EVALS:
                break
            
            if abs(f2 - f1) < 1e-30:
                break
            
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x3):
                break
            
            f3 = eval_f(x3)
            if not math.isfinite(f3):
                break
            
            if abs(f3) < abs(best_fx):
                best_x, best_fx = x3, f3
                if is_good(x3, f3):
                    return x3
            
            x1, f1 = x2, f2
            x2, f2 = x3, f3
        
        return best_x
    
    # Main logic
    x = float(x0)
    fx = eval_f(x)
    
    if is_good(x, fx):
        return x
    
    best_x = x
    best_fx = fx
    best_abs = abs(fx)
    
    # Step 1: Try to find a bracket
    bracket = find_bracket(x, fx)
    
    if bracket is not None:
        a, fa, b, fb = bracket
        if is_good(a, fa):
            return a
        if is_good(b, fb):
            return b
        
        # Use Illinois method on the bracket
        result = illinois_method(a, fa, b, fb, max_iter=400)
        if result is not None:
            fr = eval_f(result)
            if is_good(result, fr):
                return result
            if abs(fr) < best_abs:
                best_x, best_fx, best_abs = result, fr, abs(fr)
    
    # Step 2: If no bracket or Illinois didn't converge, try local search
    if not is_good(best_x, best_fx):
        # Try a few local perturbations to find a sign change
        scale = max(1e-8, abs(x0) * 1e-8)
        for mult in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
            if check_time() or eval_count[0] >= MAX_EVALS:
                break
            
            h = mult * scale
            for direction in [1, -1]:
                if check_time() or eval_count[0] >= MAX_EVALS:
                    break
                
                x_try = x0 + direction * h
                f_try = eval_f(x_try)
                
                if not math.isfinite(f_try):
                    continue
                
                if is_good(x_try, f_try):
                    return x_try
                
                if abs(f_try) < best_abs:
                    best_x, best_fx, best_abs = x_try, f_try, abs(f_try)
                
                # Check for sign change with x0
                if fx * f_try < 0:
                    a = x0
                    fa = fx
                    b = x_try
                    fb = f_try
                    result = illinois_method(a, fa, b, fb, max_iter=200)
                    if result is not None:
                        fr = eval_f(result)
                        if is_good(result, fr):
                            return result
                        if abs(fr) < best_abs:
                            best_x, best_fx, best_abs = result, fr, abs(fr)
                    break
    
    # Step 3: Polish
    if not is_good(best_x, best_fx):
        result = polish(best_x, best_fx)
        if is_good(result, eval_f(result)):
            return result
        if abs(eval_f(result)) < abs(best_fx):
            best_x = result
    
    return best_x

# EVOLVE-BLOCK-END
