"""Root finding: Ill-conditioned - Illinois Method with Adaptive Bracketing and Newton Polish."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Strategy:
    1. Adaptive step-doubling to find an initial bracket [a, b] with sign change.
    2. Illinois Method to converge to the root (robust for ill-conditioned functions).
    3. Newton Polish (FD derivative) to drive |f(x)| below 1e-8 efficiently.
    4. Fallback to best point if no bracket found.
    """
    start_time = time.time()
    MAX_WALL = 110.0  # Safety margin for 120s limit
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
        val = f(x)
        if not math.isfinite(val):
            val = 0.0
        
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        
        return val

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def check_budget():
        return eval_count[0] < MAX_EVALS and (time.time() - start_time < MAX_WALL)

    # --- Strategy 1: Find a Bracket ---
    def find_bracket(x0):
        """
        Searches for a bracket [a, b] such that f(a) and f(b) have opposite signs.
        Uses step-doubling in both directions from x0.
        Returns (a, b, fa, fb) or None.
        """
        if not check_budget():
            return None
            
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return (x, x, fx, fx) # Degenerate bracket, handled by caller
        
        h = 1e-3 * max(1.0, abs(x))
        
        # Search to the right
        a = x
        fa = fx
        b = x + h
        fb = eval_f(b)
        
        # If we found a sign change or a root, return
        if is_good(b, fb):
            return (a, b, fa, fb)
        if fa * fb < 0:
            return (a, b, fa, fb)

        # Expand to the right
        for _ in range(50):
            if not check_budget():
                break
            a = b
            fa = fb
            h *= 2.0
            b = a + h
            fb = eval_f(b)
            if is_good(b, fb):
                return (a, b, fa, fb)
            if fa * fb < 0:
                return (a, b, fa, fb)
            if not math.isfinite(fb):
                break

        # Search to the left
        a = x
        fa = fx
        b = x - h
        fb = eval_f(b)
        
        if is_good(b, fb):
            return (a, b, fa, fb)
        if fa * fb < 0:
            return (a, b, fa, fb)

        # Expand to the left
        for _ in range(50):
            if not check_budget():
                break
            a = b
            fa = fb
            h *= 2.0
            b = a - h
            fb = eval_f(b)
            if is_good(b, fb):
                return (a, b, fa, fb)
            if fa * fb < 0:
                return (a, b, fa, fb)
            if not math.isfinite(fb):
                break

        return None

    # --- Strategy 2: Illinois Method ---
    def illinois_method(a, b, fa, fb, max_iter=100):
        """
        Illinois method: A bracketing method that uses secant steps but shrinks
        the larger sub-interval by a fixed factor if the new point is on the same
        side. This guarantees linear convergence and handles ill-conditioning better
        than standard secant.
        """
        if not check_budget():
            return None

        SHRINK = 0.5
        # Ensure a < b for clarity, though math works either way
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        for _ in range(max_iter):
            if not check_budget():
                break
            
            # Check convergence criteria
            if is_good(a, fa):
                return a
            if is_good(b, fb):
                return b
            
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                # Interval is tiny, return the midpoint or the one with smaller |f|
                return a if abs(fa) < abs(fb) else b

            # Illinois Update
            # Compute secant point
            if fa == fb:
                # Parallel, bisect
                c = 0.5 * (a + b)
            else:
                c = a - fa * (b - a) / (fb - fa)
            
            if not math.isfinite(c):
                c = 0.5 * (a + b)
            
            fc = eval_f(c)
            if is_good(c, fc):
                return c

            # Determine which side the root is on
            if fa * fc < 0:
                # Root is in [a, c]
                # We replace b with c.
                # Since [a, c] is the smaller interval (usually), we shrink the larger one [c, b]
                # But Illinois logic: 
                # If we keep [a, c], we shrink the old [c, b] part? 
                # Standard Illinois: 
                # If f(a) and f(c) have same sign -> shrink [a, c]
                # If f(c) and f(b) have same sign -> shrink [c, b]
                
                # Here fa * fc < 0 means f(a) and f(c) have DIFFERENT signs.
                # So the root is in [a, c].
                # We discard [c, b].
                # To maintain the "Illinois" property, we shrink the interval that was discarded? 
                # No, Illinois shrinks the interval that *contains the previous point* that didn't cross.
                
                # Let's stick to the standard definition:
                # 1. Calculate c.
                # 2. If f(a) and f(c) have same sign, replace a with c and shrink [a, c] (the new interval) ? 
                #    Actually, Illinois replaces the point that is on the same side as the previous one.
                
                # Correct Illinois Logic:
                # We have [a, b]. We compute c.
                # If f(a)*f(c) < 0: Root in [a, c]. Set b = c, fb = fc. 
                #    But we must shrink the interval [c, old_b] ? No, we shrink the interval that we are *keeping*? 
                #    No, Illinois shrinks the *larger* of the two sub-intervals? 
                #    Actually, Illinois is defined as:
                #    If f(a) and f(c) have same sign: 
                #       a = c
                #       fa = fc
                #       b = a + SHRINK * (b - a)  # Shrink the interval from the new a towards old b?
                #       # Wait, if we set a=c, the interval is [c, b]. We shrink it from the left?
                #       # Standard: b_new = a_new + SHRINK * (b_old - a_new)
                #    If f(c) and f(b) have same sign:
                #       b = c
                #       fb = fc
                #       a = b - SHRINK * (b - a) # Shrink the interval from the new b towards old a?
                #       # Standard: a_new = b_new - SHRINK * (b_new - a_old)
                
                # Let's implement this standard variant.
                
                # Case 1: f(a) and f(c) have DIFFERENT signs (fa * fc < 0)
                # Root is in [a, c]. 
                # We set b = c.
                # We shrink the interval [a, c] from the right? 
                # The "other" point was b. The interval [c, b] is discarded.
                # Illinois shrinks the interval that *remains* if the step was "bad"?
                # Actually, the definition is:
                # If f(a) and f(c) have same sign: 
                #   a <- c
                #   b <- a + SHRINK*(b-a)  (b is updated relative to new a)
                # If f(b) and f(c) have same sign:
                #   b <- c
                #   a <- b - SHRINK*(b-a) (a is updated relative to new b)
                
                # In our case fa * fc < 0. So f(a) and f(c) have DIFFERENT signs.
                # This implies f(b) and f(c) have SAME sign (since f(a) and f(b) had different signs initially).
                # So we are in the "f(b) and f(c) have same sign" case.
                b = c
                fb = fc
                # Shrink the interval from the left (a) towards the right (new b)?
                # No, the formula is: a_new = b_new - SHRINK * (b_new - a_old)
                # This moves a closer to b.
                a = b - SHRINK * (b - a)
                # Note: we don't evaluate f(a) here, we keep fa? 
                # No, in Illinois, we do NOT evaluate f at the new shrunk endpoint. 
                # We keep the OLD fa. 
                # Wait, if we move a, fa is no longer f(a). 
                # Standard Illinois: 
                # We keep the function value of the point that was NOT replaced by the secant step?
                # Let's trace:
                # Start: a, b.
                # Calc c.
                # If f(b) and f(c) same sign:
                #   b_new = c
                #   a_new = b_new - SHRINK * (b_new - a_old)
                #   We keep fa_old. 
                #   We discard fb_old (since b changed to c, and fc is the new fb).
                #   So fa remains fa_old. fb becomes fc.
                pass # Logic handled below in unified way

            elif fc * fb < 0:
                # Root in [c, b].
                # f(a) and f(c) have SAME sign.
                a = c
                fa = fc
                # Shrink interval from right (b) towards left (new a)?
                # Formula: b_new = a_new + SHRINK * (b_old - a_new)
                b = a + SHRINK * (b - a)
                # fa becomes fc. fb remains fb_old.
            else:
                # c is a root or numerical issue
                if fc == 0:
                    return c
                # Bisect as fallback
                c = 0.5 * (a + b)
                fc = eval_f(c)
                if is_good(c, fc):
                    return c
                if fa * fc < 0:
                    b = c
                    fb = fc
                    a = b - SHRINK * (b - a)
                else:
                    a = c
                    fa = fc
                    b = a + SHRINK * (b - a)
            
            # We must be careful: 
            # In the branch `fa * fc < 0`, we set `b = c` and `fb = fc`. 
            # Then we update `a`. 
            # The value `fa` remains the old `fa`. 
            # This is correct for Illinois.
            
            # Check if a and b are too close
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                return a if abs(fa) < abs(fb) else b

        # Return best of the two
        return a if abs(fa) < abs(fb) else b

    # --- Strategy 3: Newton Polish ---
    def polish_root(x_start, fx_start=None, max_iter=20):
        """High-precision Newton polish."""
        if not check_budget():
            return None
        x = float(x_start)
        if fx_start is not None:
            fx = float(fx_start)
        else:
            fx = eval_f(x)
            if is_good(x, fx):
                return x
        
        for _ in range(max_iter):
            if not check_budget():
                break
            if abs(fx) <= TARGET_TOL:
                return x
            
            # Adaptive h for derivative
            h = 1e-6 * max(1.0, abs(x))
            if h < 1e-12:
                h = 1e-12
            
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
            
            # Limit step size
            max_step = 0.5 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * (1 if dx > 0 else -1)
            
            x_new = x - dx
            if not math.isfinite(x_new):
                break
            
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            
            # Damping
            if abs(fx_new) >= abs(fx):
                improved = False
                for damp in [0.5, 0.25, 0.1]:
                    if not check_budget():
                        break
                    x_d = x - damp * dx
                    fx_d = eval_f(x_d)
                    if is_good(x_d, fx_d):
                        return x_d
                    if abs(fx_d) < abs(fx):
                        x, fx = x_d, fx_d
                        improved = True
                        break
                if not improved:
                    break
            else:
                x, fx = x_new, fx_new
            
            if abs(dx) < 1e-14 * max(1.0, abs(x)):
                break
        
        return x if abs(fx) <= TARGET_TOL else None

    # === MAIN EXECUTION ===
    
    # 1. Try to find a bracket
    bracket = find_bracket(x0)
    
    if bracket is not None:
        a, b, fa, fb = bracket
        # If we found a root exactly during bracketing
        if is_good(a, fa):
            return a
        if is_good(b, fb):
            return b
        
        # 2. Apply Illinois Method
        root_candidate = illinois_method(a, b, fa, fb)
        
        if root_candidate is not None:
            # 3. Polish the result
            polished = polish_root(root_candidate)
            if polished is not None:
                return polished
            return root_candidate
    
    # 4. Fallback: No bracket found. Use best point found so far.
    # Try to polish the best point.
    polished = polish_root(best_x[0])
    if polished is not None:
        return polished
    
    return best_x[0]

# EVOLVE-BLOCK-END
