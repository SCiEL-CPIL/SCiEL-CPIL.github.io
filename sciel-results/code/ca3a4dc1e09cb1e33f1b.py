import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using an adaptive strategy:
    1. Try Newton's method first (fastest for smooth polynomials).
    2. If Newton fails (derivative ~0, divergence, or leaving bracket), 
       switch to Secant method.
    3. If Secant fails or oscillates, fall back to Bisection within a maintained bracket.
    4. Bracket is established initially via geometric expansion if needed.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    try:
        x0 = float(x0)
    except (TypeError, ValueError):
        return 0.0
        
    if not math.isfinite(x0):
        return 0.0

    # 1. Evaluate at x0
    try:
        fx = f(x0)
    except Exception:
        return 0.0
        
    if not math.isfinite(fx):
        return 0.0
        
    if abs(fx) <= 1e-8:
        return x0

    # 2. Establish an initial bracket [a, b] containing a root.
    # This is crucial for the bisection fallback and for validating Newton steps.
    a = x0
    fa = fx
    b = None
    fb = None
    
    step = 1e-2
    max_expansions = 30  # Reduced slightly, Newton is usually fast if bracket exists
    
    for i in range(max_expansions):
        if time.time() > deadline:
            return a if abs(fa) <= 1e-8 else x0
            
        # Try positive direction
        b_pos = a + step
        if math.isfinite(b_pos):
            try:
                fb_pos = f(b_pos)
            except Exception:
                fb_pos = None
            if fb_pos is not None and math.isfinite(fb_pos):
                if abs(fb_pos) <= 1e-8:
                    return b_pos
                if fa * fb_pos <= 0:
                    b = b_pos
                    fb = fb_pos
                    break
        
        # Try negative direction
        b_neg = a - step
        if math.isfinite(b_neg):
            try:
                fb_neg = f(b_neg)
            except Exception:
                fb_neg = None
            if fb_neg is not None and math.isfinite(fb_neg):
                if abs(fb_neg) <= 1e-8:
                    return b_neg
                if fa * fb_neg <= 0:
                    b = b_neg
                    fb = fb_neg
                    break
        
        step *= 2.0
        if step > 1e10:
            break

    if b is None:
        # If no bracket found, we are in trouble. Return best effort.
        # For the task constraints, a root is expected nearby.
        return x0

    # 3. Adaptive Iteration
    # State:
    # x: current estimate
    # fx: f(x)
    # a, b, fa, fb: current bracket
    # prev_x, prev_fx: for secant
    
    x = x0
    fx = fa
    prev_x = None
    prev_fx = None
    
    # Determine initial direction for secant if needed later
    # We have a, b. Let's set x to the midpoint? No, keep x=x0.
    # But we need a "previous" point for secant. 
    # If x0 is not a or b, we can use the other bracket endpoint?
    # Actually, let's just start Newton. If it fails, we use Secant with x and the bracket endpoint closer to x?
    # Or simpler: If Newton fails, switch to Bisection immediately? 
    # No, Secant is faster than Bisection.
    # Let's initialize secant history with x and the bracket endpoint that is 'better' or just a?
    # Let's use x and a (or b) for secant initialization if Newton fails.
    
    # Let's refine: 
    # If x0 is strictly inside (a,b), we have 3 points: a, x0, b.
    # We can try Newton first.
    
    max_iter = 100
    for i in range(max_iter):
        if time.time() > deadline:
            # Return best point in bracket
            if abs(fa) < abs(fb):
                return a
            else:
                return b

        if abs(fx) <= 1e-8:
            return x

        # Check if x is still in bracket
        # If x leaves bracket, Newton is bad. Fall back.
        
        # --- TRY NEWTON ---
        # Estimate derivative using finite difference
        # h = max(1e-8, abs(x) * 1e-8)
        # f'(x) ~ (f(x+h) - f(x)) / h
        # This costs 1 extra call. 
        # Is it worth it? Newton converges quadratically. Secant is superlinear (1.618).
        # For polynomials, Newton is very fast.
        # However, the finite difference derivative might be inaccurate if h is too small/large.
        # Alternative: Use Secant directly? Secant doesn't need derivative.
        # Let's compare: 
        # Newton: 1 call for f, 1 call for f(x+h). Total 2 calls/step.
        # Secant: 1 call for f(x_new). Uses previous f. Total 1 call/step (amortized).
        # But Newton steps are much shorter.
        
        # Let's try a hybrid:
        # 1. Compute Newton step.
        # 2. If Newton step is safe (inside bracket, not too small), take it.
        # 3. Else, take Secant step.
        # 4. Else, take Bisection step.
        
        newton_valid = False
        newton_x = None
        
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(x))
        xh = x + h
        try:
            fxh = f(xh)
        except Exception:
            fxh = None
        
        if fxh is not None and math.isfinite(fxh):
            dfx = (fxh - fx) / h
            if abs(dfx) > 1e-12:
                newton_x = x - fx / dfx
                if math.isfinite(newton_x):
                    # Check if newton_x is inside bracket
                    min_ab = min(a, b)
                    max_ab = max(a, b)
                    # Allow slight tolerance outside bracket? No, keep it strict for safety.
                    # If it's very close, it's fine.
                    if min_ab <= newton_x <= max_ab:
                        newton_valid = True
                    else:
                        # If Newton steps outside, it's likely diverging or root is far.
                        # Don't use it.
                        pass
        
        if newton_valid:
            x = newton_x
            try:
                fx = f(x)
            except Exception:
                fx = float('inf')
                break # Should not happen if finite
            if not math.isfinite(fx):
                break
            # Update bracket if x is inside
            if a < x < b:
                if fa * fx <= 0:
                    b = x
                    fb = fx
                else:
                    a = x
                    fa = fx
            elif b < x < a:
                if fb * fx <= 0:
                    a = x
                    fa = fx
                else:
                    b = x
                    fb = fx
            
            # Update secant history
            prev_x = x
            prev_fx = fx
            continue

        # --- FALLBACK TO SECANT ---
        # We need two points. We have x, fx.
        # If prev_x is None, we need to establish it.
        # We can use the bracket endpoint a or b.
        # Which one is better? The one with smaller |f|?
        # Or just use a?
        # Let's use the point in the bracket that is not x (if x is a or b) or the midpoint?
        # Actually, if Newton failed, we likely don't have a good secant history.
        # Let's just use Bisection if we don't have a good secant pair.
        
        if prev_x is not None and prev_x != x:
            # Secant method
            denom = fx - prev_fx
            if abs(denom) > 1e-12:
                secant_x = x - fx * (x - prev_x) / denom
                if math.isfinite(secant_x):
                    min_ab = min(a, b)
                    max_ab = max(a, b)
                    if min_ab <= secant_x <= max_ab:
                        x = secant_x
                        try:
                            fx = f(x)
                        except Exception:
                            fx = float('inf')
                            break
                        if not math.isfinite(fx):
                            break
                        if abs(fx) <= 1e-8:
                            return x
                        # Update bracket
                        if a < x < b:
                            if fa * fx <= 0:
                                b = x
                                fb = fx
                            else:
                                a = x
                                fa = fx
                        elif b < x < a:
                            if fb * fx <= 0:
                                a = x
                                fa = fx
                            else:
                                b = x
                                fb = fx
                        prev_x = x
                        prev_fx = fx
                        continue
        
        # --- FALLBACK TO BISECTION ---
        # If Newton and Secant are not viable, bisect.
        mid = (a + b) / 2.0
        x = mid
        try:
            fx = f(x)
        except Exception:
            break
        if not math.isfinite(fx):
            break
            
        if abs(fx) <= 1e-8:
            return x

        # Update bracket
        if fa * fx <= 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx
            
        # Reset secant history to avoid using old/stale points?
        # Or keep it? If we bisect, the previous secant point might be far.
        # Let's set prev_x to the other endpoint to have a valid secant next time?
        # Actually, let's just clear it to force a fresh start or bisection again?
        # Keeping it is risky if the function is not monotonic.
        # Let's set prev_x to the midpoint of the *other* half? No.
        # Let's just clear prev_x to force Newton/Secant re-evaluation from current state?
        # If we clear prev_x, next iteration Newton is tried. If it fails, Secant fails (no prev), Bisection used.
        # This might be slow if Newton keeps failing.
        # Better: Set prev_x to the point we just bisected from?
        # Let's set prev_x to the opposite endpoint of the bracket from x.
        if a < x < b:
            # x is new b or a?
            # If fa*fx <= 0, b=x. So a is the other.
            # If fa*fx > 0, a=x. So b is the other.
            if b == x:
                prev_x = a
                prev_fx = fa
            else:
                prev_x = b
                prev_fx = fb
        elif b < x < a:
            if a == x:
                prev_x = b
                prev_fx = fb
            else:
                prev_x = a
                prev_fx = fa

    # Return best found
    if abs(fa) < abs(fb):
        return a
    else:
        return b
