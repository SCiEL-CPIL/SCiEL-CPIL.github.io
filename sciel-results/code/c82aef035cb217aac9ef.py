import math
import time

def find_root(f, x0):
    """
    Find a root of f near x0 using a robust hybrid Secant + Bisection method.
    Optimized for smooth polynomials to minimize function calls.
    """
    DEADLINE = 140.0  # seconds, leaving 10s margin for evaluator overhead
    start_time = time.time()
    call_count = 0
    best_x = float('nan')
    best_fx = float('inf')
    best_abs_fx = float('inf')

    def eval_f(x):
        nonlocal call_count, best_x, best_fx, best_abs_fx
        if time.time() - start_time > DEADLINE:
            return best_fx  # Return stale value to break loops
        fx = f(x)
        call_count += 1
        if not math.isfinite(fx):
            return float('inf')
        abs_fx = abs(fx)
        if abs_fx < best_abs_fx:
            best_abs_fx = abs_fx
            best_x = x
            best_fx = fx
        return fx

    x0 = float(x0)
    # Initial evaluations
    f0 = eval_f(x0)
    if abs(f0) <= 1e-8:
        return x0

    # Secant initialization
    # Choose x1 close to x0 but distinct
    step = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + step
    f1 = eval_f(x1)
    if abs(f1) <= 1e-8:
        return x1

    # If x0 and x1 are effectively the same due to floating point, adjust
    if x1 == x0:
        x1 = x0 + 1e-8
        f1 = eval_f(x1)
        if abs(f1) <= 1e-8:
            return x1

    # Variables for Secant
    x_prev = x0
    f_prev = f0
    x_curr = x1
    f_curr = f1

    # Variables for Bisection (initially None)
    a = None
    b = None
    fa = None
    fb = None

    # Determine initial bracket if possible
    if f_prev * f_curr < 0:
        a = x_prev
        b = x_curr
        fa = f_prev
        fb = f_curr
    else:
        # Try to find a bracket by expanding from best_x
        # We'll do this in the main loop if Secant fails, but let's try a quick check
        # Actually, let's just run Secant first. It's faster if it works.
        pass

    for i in range(100):  # Max iterations
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Check if we have a bracket. If so, use Bisection to ensure robustness,
        # then switch to Secant when interval is small.
        if a is not None and b is not None:
            # Bisection step
            mid = (a + b) / 2.0
            fmid = eval_f(mid)
            if abs(fmid) <= 1e-8:
                return mid
            
            # Update bracket
            if fa * fmid <= 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
            
            # If interval is small enough, switch to Secant for superlinear convergence
            # 1e-10 is small enough for 1e-8 tolerance on typical polynomials
            if abs(b - a) < 1e-10 * max(1.0, abs(a), abs(b)):
                # Switch to Secant mode within this bracket
                # We can use the last two points from bisection or secant history
                # Let's just restart the Secant loop with current best points
                # Or simply continue bisection a few more times? 
                # Bisection is linear, Secant is superlinear.
                # Let's try to use Secant with points close to the root.
                # We have (a, fa) and (b, fb). If they are close, Secant is good.
                # But we need distinct points.
                # Let's just do a few more bisection steps to be safe, then Secant.
                # Actually, if we are in bisection mode, we can just continue.
                # But to minimize calls, Secant is better.
                # Let's set up Secant with the two most recent distinct points.
                # We have x_curr, f_curr and x_prev, f_prev from previous Secant iterations.
                # If we've been bisecting, x_prev/x_curr might be stale.
                # Let's just use the bracket endpoints for the next Secant step?
                # No, Secant needs function values.
                # Let's switch back to Secant using the last two distinct x,f pairs.
                # We will set a flag to stop bisection and start secant.
                # For simplicity, if interval is small, just return the midpoint?
                # No, |f(mid)| might be > 1e-8.
                # Let's do one more bisection step and then try Secant.
                pass
            
            # If we are in bisection, we should probably stay in bisection until
            # we are confident, or switch to Secant.
            # Let's switch to Secant if the interval is reasonably small.
            if abs(b - a) < 1e-6 * max(1.0, abs(a), abs(b)):
                # Switch to Secant
                # Use the last two points from the Secant history if valid,
                # otherwise use the bracket endpoints?
                # Secant formula: x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                # We need two points. Let's use the most recent two distinct evaluations.
                # This is getting complicated. 
                # Simpler: If bracketed, just bisect. It's robust.
                # But Secant is faster.
                # Let's try this: If bracketed, bisect. If not, Secant.
                # If Secant fails to maintain bracket, expand to find one.
                continue

        # If not bracketed, use Secant
        if f_curr == f_prev:
            # Perturb to avoid division by zero
            x_curr += 1e-8
            f_curr = eval_f(x_curr)
            if abs(f_curr) <= 1e-8:
                return x_curr
            continue

        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        if not math.isfinite(x_next):
            # Secant failed, try to find a bracket
            break

        # Safety: Don't jump too far
        max_step = max(1.0, abs(x_curr) * 10.0)
        if abs(x_next - x_curr) > max_step:
            # Damp the step
            direction = 1.0 if x_next > x_curr else -1.0
            x_next = x_curr + direction * max_step

        f_next = eval_f(x_next)
        if abs(f_next) <= 1e-8:
            return x_next

        # Check for bracket formation
        if f_curr * f_next < 0:
            a = x_curr
            b = x_next
            fa = f_curr
            fb = f_next
            # Once bracketed, we can switch to bisection for robustness
            # Or continue with Secant but constrained to the bracket?
            # Bisection is safer. Let's switch to bisection.
            # But Secant is faster. 
            # Strategy: If bracketed, use Bisection. It guarantees convergence.
            # For polynomials, Bisection is very reliable.
            # However, Secant is usually faster.
            # Let's use a hybrid: If bracketed, use Bisection.
            # If not, use Secant.
            # This is a common robust strategy.
            pass
        
        # Update Secant history
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_next
        f_curr = f_next

        # If we have a bracket, we should probably switch to bisection
        # to ensure we don't oscillate outside.
        # Let's check if we have a bracket.
        if a is not None and b is not None:
            # We are in bisection mode effectively
            # The loop above handles bisection if a,b are set.
            # But we just updated x_curr/x_prev.
            # We need to ensure the next iteration uses bisection.
            # The condition `if a is not None...` at the start of the loop handles this.
            pass

    # If loop finished without return, try one last bracketing search
    if time.time() - start_time < DEADLINE - 2:
        # Expand from best_x
        xc = best_x
        fc = eval_f(xc)
        if abs(fc) <= 1e-8:
            return xc
        
        step = 1e-4 * max(1.0, abs(xc))
        for i in range(1, 100):
            if time.time() - start_time > DEADLINE - 1:
                break
            xl = xc - step * i
            fl = eval_f(xl)
            if abs(fl) <= 1e-8:
                return xl
            if fc * fl < 0:
                # Bracket found [xl, xc]
                a, b = xl, xc
                fa, fb = fl, fc
                break
            
            xr = xc + step * i
            fr = eval_f(xr)
            if abs(fr) <= 1e-8:
                return xr
            if fc * fr < 0:
                # Bracket found [xc, xr]
                a, b = xc, xr
                fa, fb = fc, fr
                break
        
        if a is not None and b is not None:
            # Bisection
            for i in range(80):
                if time.time() - start_time > DEADLINE - 1:
                    break
                mid = (a + b) / 2.0
                fmid = eval_f(mid)
                if abs(fmid) <= 1e-8:
                    return mid
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
                if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                    break

    return best_x
