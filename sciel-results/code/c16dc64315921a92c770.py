import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    calls = 0
    max_calls = 480
    best_x = float('nan')
    best_val = float('inf')
    
    def _f(x):
        nonlocal calls, best_x, best_val
        if calls >= max_calls:
            return float('nan')
        if time.time() > deadline:
            return float('nan')
        calls += 1
        try:
            val = f(x)
            if math.isfinite(val):
                av = abs(val)
                if av < best_val:
                    best_val = av
                    best_x = x
                if av <= 1e-8:
                    return val
            return val
        except:
            return float('nan')

    def time_ok():
        return time.time() < deadline

    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x if not math.isnan(best_x) else x
    if abs(fx) <= 1e-8: return x

    # 1. Find Bracket using multi-scale grid search
    # Strategy: Evaluate at exponentially spaced points to find a sign change
    # We check x0, then x0 +/- delta, x0 +/- 2*delta, ... with increasing delta
    # This is more robust than just expanding one direction at a time if the
    # function oscillates rapidly or the root is far away but the local slope is small.
    
    a, b, fa, fb = None, None, None, None
    found_bracket = False
    
    # Determine base scale
    base_step = 1.0
    if abs(x) > 1e6: base_step = 1e6
    elif abs(x) > 1e3: base_step = 1e3
    elif abs(x) > 1: base_step = 1.0
    else: base_step = 0.1
    
    # We will test points: x, x +/- step, x +/- 2*step, x +/- 4*step...
    # But to save calls, we interleave positive and negative or just go one way?
    # The current program goes positive then negative.
    # The "inspiration" suggests a grid. Let's do a hybrid:
    # Check a few close points, then expand.
    
    # Actually, the current program's expansion is decent. 
    # The "Global multi-scale grid search" idea implies checking many points.
    # Let's stick to the efficient expansion but ensure we cover both sides reasonably.
    
    # Let's try a slightly more aggressive initial probe if x0 is 0 or small.
    # If x0 is 0, base_step is 0.1.
    
    # Phase 1: Local probe (small steps)
    # Check x +/- 0.01, x +/- 0.05, x +/- 0.1 etc?
    # The current code does:
    # step = base_step * 0.1
    # loop: xp = x + step, step *= 1.5
    
    # Let's refine the bracket finding to be a bit more thorough with a few fixed multipliers
    # to catch sign changes that might be missed by pure geometric progression if the
    # function is very oscillatory (e.g. sin(x^2)).
    # However, sin(x^2) roots get closer together. Geometric progression might skip over
    # a root if the step is too large.
    # To handle this, we should ensure steps are small enough relative to the frequency.
    # Since we don't know the frequency, we can't be perfect.
    # But we can do a "dense" local search first.
    
    # Let's try:
    # 1. Check x
    # 2. Check x +/- 0.01, 0.05, 0.1, 0.5, 1.0 (relative to base_step scale?)
    
    # Let's stick to the proven geometric expansion but start with a smaller step
    # and do a few more iterations if possible within budget.
    
    # Current: 20 iterations positive, 20 negative. Step *= 1.5.
    # 20 * 1.5^20 is huge.
    
    # Let's try a two-phase bracket:
    # Phase A: Fine grid near x0
    # Phase B: Coarse grid further out
    
    # Fine grid:
    fine_steps = [0.01, 0.05, 0.1, 0.5, 1.0]
    # Scale fine steps by base_step? No, base_step is already scaled.
    # If base_step is 1.0, fine steps are 0.01, 0.05...
    # If base_step is 1e-3, fine steps should be smaller.
    
    # Let's just use the geometric expansion but start tighter.
    
    # Positive direction
    step = base_step * 0.05  # Start tighter
    prev_x, prev_f = x, fx
    for i in range(1, 25):
        if not time_ok() or calls >= max_calls - 20: break
        xp = x + step
        fp = _f(xp)
        if math.isnan(fp): break
        if abs(fp) <= 1e-8: return xp
        if prev_f * fp < 0:
            a, b, fa, fb = prev_x, xp, prev_f, fp
            found_bracket = True
            break
        prev_x, prev_f = xp, fp
        step *= 1.5
        
    if not found_bracket:
        # Negative direction
        step = base_step * 0.05
        prev_x, prev_f = x, fx
        for i in range(1, 25):
            if not time_ok() or calls >= max_calls - 20: break
            xn = x - step
            fn = _f(xn)
            if math.isnan(fn): break
            if abs(fn) <= 1e-8: return xn
            if prev_f * fn < 0:
                a, b, fa, fb = xn, x, fn, fx
                found_bracket = True
                break
            prev_x, prev_f = xn, fn
            step *= 1.5

    if found_bracket:
        # 2. Refine using Illinois Method (Bisection + Secant hybrid)
        # Illinois is more robust than pure Secant for oscillatory functions
        # and faster than pure Bisection.
        
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        for i in range(100):
            if calls >= max_calls or not time_ok(): break
            if abs(fa) <= 1e-8: return a
            if abs(fb) <= 1e-8: return b
            
            # Illinois Step
            # Calculate secant point
            if abs(fa - fb) > 1e-30:
                m = (a * fb - b * fa) / (fb - fa)
            else:
                m = (a + b) / 2
                
            # If secant point is outside bracket, use bisection
            if not (a < m < b):
                m = (a + b) / 2
            else:
                # Illinois modification:
                # If the new point is on the same side as the previous point (or generally),
                # shrink the farther endpoint.
                # Standard Illinois:
                # If fm has same sign as fa, replace a with m, and set fa = fa/2
                # If fm has same sign as fb, replace b with m, and set fb = fb/2
                fm = _f(m)
                if math.isnan(fm): break
                if abs(fm) <= 1e-8: return m
                
                if fa * fm < 0:
                    # Root in [a, m]
                    b, fb = m, fm
                    # Illinois: shrink the other end?
                    # Actually, standard Illinois shrinks the function value of the
                    # point that was just discarded or kept?
                    # Wikipedia: "Illinois method... if the new point is on the same side
                    # as the previous point, halve the function value of the other point."
                    # Let's implement the standard halving to force convergence if secant stalls.
                    # If we replaced b, we halve fa? Or fa?
                    # If fa*fm < 0, root is in [a,m]. b is discarded.
                    # We halve fa? No, we halve the function value of the point that is
                    # NOT being replaced?
                    # Let's use the simpler "Shrinking" variant if secant is slow.
                    # Or just stick to the standard Bisection-Secant guard which is very robust.
                    # The current program's "Guarded Secant" is effectively Illinois-like.
                    pass
                else:
                    a, fa = m, fm

            # To be safe and fast, let's just use the robust Guarded Secant from the current program
            # which is already performing well (1.7046).
            # The main improvement opportunity is in the bracket finding.
            # Let's revert refinement to the known good one.
            
        # Re-do refinement with the known good logic
        # Reset a,b,fa,fb? No, they are still valid.
        
        # Let's just run the refinement loop properly.
        # I will copy the refinement from the current program as it is solid.
        
        # Re-initialize for clarity in the loop below
        # (a,b,fa,fb are already set)
        
        for i in range(100):
            if calls >= max_calls or not time_ok(): break
            if abs(fa) <= 1e-8: return a
            if abs(fb) <= 1e-8: return b
            
            if fa == fb:
                m = (a+b)/2
            else:
                m = (a*fb - b*fa) / (fb - fa)
                
            if a < m < b:
                fm = _f(m)
                if math.isnan(fm): break
                if abs(fm) <= 1e-8: return m
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            else:
                m = (a+b)/2
                fm = _f(m)
                if math.isnan(fm): break
                if abs(fm) <= 1e-8: return m
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            
        if abs(fa) <= 1e-8: return a
        if abs(fb) <= 1e-8: return b
        return a if abs(fa) < abs(fb) else b

    # No bracket found
    # Fallback: return best_x
    if not math.isnan(best_x):
        return best_x
    return x
