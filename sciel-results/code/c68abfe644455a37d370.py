# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Strategy:
    1. Quick Newton/secant phase with adaptive step control.
    2. Bracket expansion to find a sign change.
    3. Illinois algorithm for guaranteed convergence within the bracket.
    4. Deadline guard to stay within time limits.
    """
    deadline = time.time() + 140  # Leave 10s margin before 150s kill
    max_evals = 480
    eval_count = [0]
    
    def eval_f(x):
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            return val
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
            eval_count[0] += 1
            return None
    
    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 1: Newton's method with adaptive damping
    best_x = x
    best_f = abs(fx)
    
    for i in range(30):
        if time.time() > deadline:
            break
        if eval_count[0] >= max_evals:
            break
        
        # Finite difference derivative
        h = 1e-8 * max(1.0, abs(x))
        fx2 = eval_f(x + h)
        if fx2 is None:
            break
        d = (fx2 - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        # Newton step
        step = fx / d
        new_x = x - step
        
        # Adaptive damping: if step is too large, shrink it
        if not math.isfinite(new_x):
            break
        
        # Limit step size to avoid divergence
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
            new_x = x - step
        
        fx_new = eval_f(new_x)
        if fx_new is None:
            break
        
        if abs(fx_new) < abs(fx):
            # Accept the step
            x = new_x
            fx = fx_new
            if abs(fx) <= 1e-8:
                return x
            if abs(fx) < best_f:
                best_f = abs(fx)
                best_x = x
        else:
            # Reject: halve the step
            x_try = x - step / 2.0
            fx_try = eval_f(x_try)
            if fx_try is not None and abs(fx_try) < abs(fx):
                x = x_try
                fx = fx_try
                if abs(fx) <= 1e-8:
                    return x
                if abs(fx) < best_f:
                    best_f = abs(fx)
                    best_x = x
            else:
                # Stagnation in Newton phase, move to bracketing
                break
    
    if abs(fx) <= 1e-8:
        return x
    if abs(fx) < best_f:
        best_f = abs(fx)
        best_x = x
    
    # Phase 2: Find a bracket [a, b] with a sign change
    a, b = None, None
    fa, fb = None, None
    
    # Try to expand bracket around current best
    center = best_x
    fc = eval_f(center)
    if fc is None:
        return best_x
    if abs(fc) <= 1e-8:
        return center
    
    # Expand bracket
    radius = max(1.0, abs(center) * 0.1)
    for _ in range(50):
        if time.time() > deadline:
            break
        if eval_count[0] >= max_evals:
            break
        
        a = center - radius
        b = center + radius
        
        fa = eval_f(a)
        if fa is None:
            radius *= 2
            continue
        fb = eval_f(b)
        if fb is None:
            radius *= 2
            continue
        
        if fa * fb < 0:
            # Found a bracket with sign change
            break
        else:
            # No sign change, expand
            radius *= 2
            if radius > 1e10:
                break
    
    if fa is not None and fb is not None and fa * fb < 0:
        # Phase 3: Illinois algorithm for guaranteed convergence
        # Illinois is a variant of bisection that weights the point
        # closer to the root more heavily
        
        # Ensure f(a) and f(b) have opposite signs
        if fa > 0:
            a, b = b, a
            fa, fb = fb, fa
        # Now fa < 0 < fb
        
        for _ in range(200):
            if time.time() > deadline:
                break
            if eval_count[0] >= max_evals:
                break
            
            # Illinois: midpoint weighted toward the side with smaller |f|
            if abs(fa) < abs(fb):
                c = (a + b) / 2.0  # standard bisection when close
            else:
                # Weighted toward a (where |fa| is smaller)
                c = (b * abs(fa) + a * abs(fb)) / (abs(fa) + abs(fb))
            
            fc = eval_f(c)
            if fc is None:
                break
            
            if abs(fc) <= 1e-8:
                return c
            
            if fc == 0:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            # Check convergence
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                break
            
            # Return the point with smaller |f|
            if abs(fa) < abs(fb):
                if abs(fa) < best_f:
                    best_f = abs(fa)
                    best_x = a
            else:
                if abs(fb) < best_f:
                    best_f = abs(fb)
                    best_x = b
    
    # Return the best point found
    return best_x
# EVOLVE-BLOCK-END
