import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses Illinois Method (accelerated bisection) with Newton acceleration.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    max_evals = 490  # Safety margin for 500 limit
    TOL = 1e-8
    
    eval_count = 0
    best_x = float(x0)
    best_fabs = float('inf')
    
    def eval_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            return None
        if time.time() > deadline:
            return None
        eval_count += 1
        try:
            fx = f(x)
        except Exception:
            return None
        if math.isfinite(fx):
            fabs = abs(fx)
            if fabs < best_fabs:
                best_fabs = fabs
                best_x = x
            return fx
        return None

    def time_exceeded():
        return time.time() > deadline

    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x

    # Initialize secant points
    # Try multiple step sizes to find a bracket or good secant pair
    h = 1e-3 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is None:
        return best_x
    if abs(fx2) <= TOL:
        return x2
    
    # If no bracket with first step, try larger steps
    if fx * fx2 > 0:
        for h_factor in [1e-2, 1e-1, 1.0, 10.0]:
            h2 = h_factor * max(1.0, abs(x))
            x3 = x + h2
            fx3 = eval_f(x3)
            if fx3 is None:
                continue
            if abs(fx3) <= TOL:
                return x3
            if fx * fx3 < 0:
                x2, fx2 = x3, fx3
                break
            elif abs(fx3) < abs(fx2):
                x2, fx2 = x3, fx3

    # State: We have (x, fx) and (x2, fx2)
    # Check for bracket
    if fx * fx2 < 0:
        a, b = x, x2
        fa, fb = fx, fx2
        
        for _ in range(300):
            if time_exceeded() or eval_count >= max_evals:
                break
            
            # Illinois-style accelerated bisection
            # If |fa| > |fb|, root is closer to b; damp fa for secant
            if abs(fa) > abs(fb):
                fa_damped = 0.5 * fa
            else:
                fb_damped = 0.5 * fb
            
            # Compute secant point using damped value
            if abs(fa) > abs(fb):
                if abs(fa_damped - fb) > 1e-30:
                    c = (a * fb - b * fa_damped) / (fb - fa_damped)
                else:
                    c = 0.5 * (a + b)
            else:
                if abs(fa - fb_damped) > 1e-30:
                    c = (a * fb_damped - b * fa) / (fb_damped - fa)
                else:
                    c = 0.5 * (a + b)
            
            # If secant is outside [a,b], use midpoint
            if not (a < c < b):
                c = 0.5 * (a + b)
            
            fc = eval_f(c)
            if fc is None:
                break
            if abs(fc) <= TOL:
                return c
            
            # Update Bracket
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
    else:
        # No bracket, use Secant with Damping (Line Search)
        x_prev = x
        fx_prev = fx
        x_curr = x2
        fx_curr = fx2
        
        for _ in range(300):
            if time_exceeded() or eval_count >= max_evals:
                break
            
            if abs(fx_curr) <= TOL:
                return x_curr
            
            # Secant step
            if abs(x_curr - x_prev) < 1e-15:
                break
            slope = (fx_curr - fx_prev) / (x_curr - x_prev)
            if abs(slope) < 1e-30:
                # Flat, perturb
                x_new = x_curr + 1e-6
            else:
                x_new = x_curr - fx_curr / slope
            
            # Line Search: Ensure |f(x_new)| < |f(x_curr)|
            # If not, bisect the step
            f_curr_abs = abs(fx_curr)
            
            # Max 20 line search iterations
            step = x_new - x_curr
            fx_new = None
            for _ls in range(20):
                fx_new = eval_f(x_curr + step)
                if fx_new is None:
                    return best_x
                if abs(fx_new) < f_curr_abs:
                    break
                # Halve the step
                step *= 0.5
            
            if time_exceeded() or fx_new is None:
                break
            
            # Check success
            if abs(fx_new) <= TOL:
                return x_curr + step
            
            # Check for new bracket
            if fx_new * fx_curr < 0:
                # Switch to Bracket Mode
                a, b = x_curr, x_curr + step
                fa, fb = fx_curr, fx_new
                # Proceed to bracket refinement with Illinois
                for _ in range(300):
                    if time_exceeded() or eval_count >= max_evals:
                        break
                    
                    if abs(fa) > abs(fb):
                        fa_damped = 0.5 * fa
                        if abs(fa_damped - fb) > 1e-30:
                            c = (a * fb - b * fa_damped) / (fb - fa_damped)
                        else:
                            c = 0.5 * (a + b)
                    else:
                        fb_damped = 0.5 * fb
                        if abs(fa - fb_damped) > 1e-30:
                            c = (a * fb_damped - b * fa) / (fb_damped - fa)
                        else:
                            c = 0.5 * (a + b)
                    
                    if not (a < c < b):
                        c = 0.5 * (a + b)
                        
                    fc = eval_f(c)
                    if fc is None:
                        break
                    if abs(fc) <= TOL:
                        return c
                    
                    if fa * fc < 0:
                        b = c
                        fb = fc
                    else:
                        a = c
                        fa = fc
                break # Exit secant loop
            
            # Update Secant Points
            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_curr + step
            fx_curr = fx_new

    # Final polish: if we're close, try a few more secant steps with tighter tolerance
    if best_fabs <= 1e-4 and eval_count < max_evals - 10:
        x_curr = best_x
        fx_curr = eval_f(x_curr)
        if fx_curr is not None and abs(fx_curr) <= TOL:
            return x_curr
        # Try secant from nearby point
        h_polish = 1e-6 * max(1.0, abs(x_curr))
        x_prev = x_curr - h_polish
        fx_prev = eval_f(x_prev)
        if fx_prev is not None:
            for _ in range(30):
                if time_exceeded() or eval_count >= max_evals:
                    break
                if abs(fx_curr) <= TOL:
                    return x_curr
                if abs(x_curr - x_prev) < 1e-15:
                    break
                slope = (fx_curr - fx_prev) / (x_curr - x_prev)
                if abs(slope) < 1e-30:
                    break
                x_new = x_curr - fx_curr / slope
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx_curr):
                    x_prev, fx_prev = x_curr, fx_curr
                    x_curr, fx_curr = x_new, fx_new
                else:
                    # Damp
                    x_new = 0.5 * (x_curr + x_new)
                    fx_new = eval_f(x_new)
                    if fx_new is None:
                        break
                    if abs(fx_new) < abs(fx_curr):
                        x_prev, fx_prev = x_curr, fx_curr
                        x_curr, fx_curr = x_new, fx_new
                    else:
                        break

    return best_x
