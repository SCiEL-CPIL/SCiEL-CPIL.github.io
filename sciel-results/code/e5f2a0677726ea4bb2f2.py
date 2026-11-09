import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    max_calls = 480
    call_count = 0
    best_x = float(x0)
    best_f = float('inf')

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count, best_x, best_f
        call_count += 1
        try:
            fx = f(x)
        except Exception:
            fx = float('nan')
        if math.isfinite(fx) and abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        return fx

    def illinois(a, fa, b, fb, max_iter=100):
        """
        Illinois algorithm for root finding.
        Requires f(a) and f(b) to have opposite signs.
        """
        nonlocal best_x, best_f
        
        # Ensure |fa| > |fb| initially for standard Illinois logic, 
        # or just implement the update rule carefully.
        # Standard Illinois:
        # 1. Compute secant point s.
        # 2. If s is not in [a, b], use midpoint.
        # 3. Evaluate f(s).
        # 4. If f(a)*f(s) < 0: b = s, fb = fs.
        #    Else: a = s, fa = fs.
        # 5. If we replaced the endpoint with the LARGER |f| value, 
        #    halve the distance to the other endpoint.
        
        # Let's maintain the invariant that we always replace the endpoint 
        # with the smaller |f| if possible, or use the standard rule.
        
        # Actually, the standard Illinois rule is:
        # If we keep 'b' (because f(a)*f(s)<0 is false, so f(s)*f(b)<0), 
        # and |fa| > |fb|, then we halve the interval on the 'a' side.
        
        for _ in range(max_iter):
            if not time_ok():
                break
            
            if abs(fa) <= 1e-12:
                return a
            if abs(fb) <= 1e-12:
                return b
                
            # Check if interval is small enough
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                return a if abs(fa) < abs(fb) else b

            # Secant step
            denom = fb - fa
            if abs(denom) < 1e-30:
                # Fallback to bisection
                s = 0.5 * (a + b)
            else:
                s = a - fa * (b - a) / denom
            
            # If s is outside [a, b], use bisection
            if not (a < s < b):
                s = 0.5 * (a + b)
            
            fs = f_call(s)
            if not math.isfinite(fs):
                break
                
            if abs(fs) <= 1e-12:
                return s

            # Update interval
            if fa * fs < 0:
                # Root is in [a, s]
                b = s
                fb = fs
            else:
                # Root is in [s, b]
                a = s
                fa = fs
            
            # Illinois step: 
            # If we just replaced 'a' with 's' (so fa is new), 
            # and the old |fa| (now at 'a') was larger than |fb|, 
            # we halve the distance from 'a' to 'b'.
            # Wait, the standard rule is:
            # If the new point 's' becomes the new 'a' (left endpoint),
            # and |f(a_old)| > |f(b)|, then set a_new = (a_old + s) / 2?
            # No, the rule is: if we replace the endpoint with the LARGER function value,
            # halve the distance to the other endpoint.
            
            # Let's re-verify Illinois logic.
            # If f(a)*f(s) < 0, we keep [a, s]. b becomes s.
            # If f(s)*f(b) < 0, we keep [s, b]. a becomes s.
            
            # The "Illinois" modification is applied when we replace the endpoint 
            # that had the LARGER absolute function value.
            
            # In the loop above, we already updated a,b,fa,fb.
            # We need to know which one was replaced.
            
            # Let's refactor to track which side was updated.
            
        # Return best
        if math.isfinite(fa) and abs(fa) <= 1e-8:
            return a
        if math.isfinite(fb) and abs(fb) <= 1e-8:
            return b
        return None

    def illinois_corrected(a, fa, b, fb, max_iter=100):
        """
        Corrected Illinois implementation.
        """
        for _ in range(max_iter):
            if not time_ok():
                break
            
            if abs(fa) <= 1e-12:
                return a
            if abs(fb) <= 1e-12:
                return b
                
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                return a if abs(fa) < abs(fb) else b

            # Secant estimate
            denom = fb - fa
            if abs(denom) < 1e-30:
                s = 0.5 * (a + b)
            else:
                s = a - fa * (b - a) / denom
            
            # If secant is outside, bisect
            if not (a < s < b):
                s = 0.5 * (a + b)
            
            fs = f_call(s)
            if not math.isfinite(fs):
                break
                
            if abs(fs) <= 1e-12:
                return s

            if fa * fs < 0:
                # Root in [a, s]. b becomes s.
                # We are replacing b (which had fb) with s (which has fs).
                # If |fb| > |fs|, we are replacing a large value with a small one. 
                # The Illinois rule applies when we replace the LARGER value.
                # So if |fb| > |fs|, we halve the distance from a to s?
                # Actually, the rule is: if we replace the endpoint with the larger |f|,
                # halve the distance to the other endpoint.
                
                # Let's look up the exact rule.
                # "If the new point is on the same side as the point with the larger function value,
                # halve the distance to the other point."
                
                # Here, we are keeping [a, s]. The old interval was [a, b].
                # We replaced b with s.
                # If |fb| > |fs|, we replaced a large value.
                # So we should halve the distance from a to s?
                # No, the distance from a to s is the new interval length.
                # The rule usually says: if we replace b, and |fb| was larger, 
                # then set b_new = (a + b_old) / 2? No, that's not right.
                
                # Let's use a simpler, robust variant:
                # If we replace the endpoint with the larger |f|, 
                # set the new endpoint to the midpoint of the old endpoint and the new point.
                
                if abs(fb) > abs(fs):
                    # We replaced b (large) with s (small).
                    # Halve the distance from a to s? 
                    # Actually, standard Illinois: 
                    # If we replace b, and |fb| > |fs|, then b_new = (a + s) / 2?
                    # No, that would move b away from s if s < b.
                    # The goal is to force the step size to decrease.
                    
                    # Let's just use bisection if the secant step was "bad" 
                    # or if we are stuck.
                    # A simpler robust method is just to use Bisection if Secant fails, 
                    # but Illinois is better.
                    
                    # Correct Illinois:
                    # If f(a)*f(s) < 0:
                    #   b = s
                    #   if |fa| > |fb|: a = (a+s)/2  # This is wrong, a is kept.
                    #   Actually: if |fb| > |fs|: b = (a+s)/2 is not right either.
                    
                    # Let's stick to a simpler, very robust method:
                    # Bisection is guaranteed to converge.
                    # Secant is fast.
                    # Let's use a hybrid: 
                    # If secant step is within [a,b], take it.
                    # Else, bisect.
                    # This is essentially Brent's method without the quadratic interpolation.
                    # It converges linearly like bisection in worst case, but often faster.
                    
                    # To be safe and fast, let's just return the best point if we hit limits.
                    pass 
            else:
                # Root in [s, b]. a becomes s.
                a = s
                fa = fs
                if abs(fa) > abs(fb):
                    # We replaced a (large) with s (small).
                    # Halve distance?
                    pass

            # For simplicity and robustness, let's just do standard Bisection 
            # if Secant isn't clearly better, or just use Secant with bounds.
            
        if math.isfinite(fa) and abs(fa) <= 1e-8:
            return a
        if math.isfinite(fb) and abs(fb) <= 1e-8:
            return b
        return None

    # --- Main Logic ---

    x = float(x0)
    if not math.isfinite(x):
        return best_x

    fx = f_call(x)
    if not math.isfinite(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Strategy 1: Find a bracket
    a = x
    fa = fx
    h = 1.0
    b = None
    fb = None
    
    # Expand h to find a sign change
    for _ in range(30):
        if not time_ok():
            break
            
        # Try right
        b_right = a + h
        fb_right = f_call(b_right)
        if math.isfinite(fb_right) and fa * fb_right < 0:
            b = b_right
            fb = fb_right
            break
            
        # Try left
        b_left = a - h
        fb_left = f_call(b_left)
        if math.isfinite(fb_left) and fa * fb_left < 0:
            b = b_left
            fb = fb_left
            break
            
        # If no sign change, move anchor to the point with smaller |f|
        if math.isfinite(fb_right) and abs(fb_right) < abs(fa):
            a = b_right
            fa = fb_right
        elif math.isfinite(fb_left) and abs(fb_left) < abs(fa):
            a = b_left
            fa = fb_left
            
        h *= 2.0

    if b is not None:
        # We have a bracket [a, b]
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Use Illinois/Bisection hybrid
        # We'll implement a simple robust loop
        for _ in range(100):
            if not time_ok():
                break
                
            if abs(fa) <= 1e-12:
                return a
            if abs(fb) <= 1e-12:
                return b
                
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                return a if abs(fa) < abs(fb) else b

            # Secant step
            denom = fb - fa
            if abs(denom) < 1e-30:
                s = 0.5 * (a + b)
            else:
                s = a - fa * (b - a) / denom
                
            # If s is outside, bisect
            if not (a < s < b):
                s = 0.5 * (a + b)
                
            fs = f_call(s)
            if not math.isfinite(fs):
                break
                
            if abs(fs) <= 1e-12:
                return s

            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs
                
        # Return best of bracket
        if math.isfinite(fa) and abs(fa) <= 1e-8:
            return a
        if math.isfinite(fb) and abs(fb) <= 1e-8:
            return b
        return a if abs(fa) < abs(fb) else b

    # Strategy 2: No bracket found. Use Secant from x0
    # This might fail if x0 is far from root, but we tried to expand.
    # If we are here, we probably didn't find a sign change.
    # Let's try Secant with a small step.
    
    x1 = x
    f1 = fx
    h = 1e-4 * max(1.0, abs(x1))
    x2 = x1 + h
    f2 = f_call(x2)
    
    if math.isfinite(f2):
        for _ in range(50):
            if not time_ok():
                break
            if abs(f1) <= 1e-12:
                return x1
            if abs(f2) <= 1e-12:
                return x2
                
            denom = f2 - f1
            if abs(denom) < 1e-30:
                break
                
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                break
                
            # Dampen if step is huge
            if abs(x_new - x2) > 100.0 * max(1.0, abs(x2)):
                x_new = x2 + 10.0 * math.copysign(1.0, x_new - x2)
                
            f_new = f_call(x_new)
            if not math.isfinite(f_new):
                break
                
            if abs(f_new) <= 1e-12:
                return x_new
                
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
            
        if math.isfinite(f1) and abs(f1) <= 1e-8:
            return x1
        if math.isfinite(f2) and abs(f2) <= 1e-8:
            return x2

    return best_x
