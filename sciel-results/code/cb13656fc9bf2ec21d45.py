"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    x = float(x0)
    best_x = x
    best_fx = None
    try:
        best_fx = f(x)
    except Exception:
        pass
    
    call_count = 0
    max_calls = 450  # stay well under 500
    
    def eval_f(val):
        nonlocal call_count
        if call_count >= max_calls:
            return None
        call_count += 1
        try:
            v = f(val)
            return v
        except Exception:
            return None
    
    # Update best
    if best_fx is not None and math.isfinite(best_fx):
        pass  # already set
    
    # Try Newton's method with safeguards
    for iteration in range(60):
        if time.time() > deadline:
            break
        
        fx = eval_f(x)
        if fx is None:
            break
        
        if not math.isfinite(fx):
            # Try to recover
            break
        
        if abs(fx) < 1e-12:
            return x
        
        if best_fx is None or abs(fx) < abs(best_fx):
            best_fx = fx
            best_x = x
        
        # Estimate derivative using central difference
        h = 1e-7 * max(1.0, abs(x))
        fx_plus = eval_f(x + h)
        fx_minus = eval_f(x - h)
        
        if fx_plus is None or fx_minus is None:
            # Fall back to forward difference
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
        else:
            d = (fx_plus - fx_minus) / (2.0 * h)
        
        if not math.isfinite(d) or abs(d) < 1e-15:
            # Derivative too small or invalid, try secant-like step
            # Use a small perturbation
            x_new = x + 1e-4 * max(1.0, abs(x))
            if x_new == x:
                break
            x = x_new
            continue
        
        # Newton step
        step = fx / d
        x_new = x - step
        
        # Safeguard: limit step size
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            x_new = x + max_step * math.copysign(1.0, step)
        
        # Check if new point is valid
        if not math.isfinite(x_new):
            break
        
        # Try the Newton step
        fx_new = eval_f(x_new)
        if fx_new is None:
            break
        
        if not math.isfinite(fx_new):
            # Step went to bad region, try smaller step
            x_new = x - step * 0.5
            if not math.isfinite(x_new):
                break
            fx_new = eval_f(x_new)
            if fx_new is None or not math.isfinite(fx_new):
                break
        
        if abs(fx_new) < abs(fx):
            # Good step, accept
            x = x_new
            fx = fx_new
            if abs(fx) < 1e-8:
                return x
        else:
            # Bad step, try bisection-like approach
            # Move halfway
            x_half = (x + x_new) * 0.5
            if x_half == x:
                break
            fx_half = eval_f(x_half)
            if fx_half is None or not math.isfinite(fx_half):
                break
            if abs(fx_half) < abs(fx):
                x = x_half
                fx = fx_half
            else:
                # Try opposite direction
                x_alt = x - step * 0.5
                if not math.isfinite(x_alt):
                    break
                fx_alt = eval_f(x_alt)
                if fx_alt is None or not math.isfinite(fx_alt):
                    break
                if abs(fx_alt) < abs(fx):
                    x = x_alt
                    fx = fx_alt
                else:
                    # Stuck, try a small random-ish perturbation
                    x = x + 1e-3 * max(1.0, abs(x)) * (1 if iteration % 2 == 0 else -1)
                    continue
        
        if abs(fx) < 1e-8:
            return x
    
    # If we still haven't found a good root, try a broader search
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x
    
    # Try geometric expansion to find a sign change, then bisection
    if time.time() < deadline and call_count < max_calls - 20:
        # Find a bracket
        a = x0
        fa = best_fx
        if fa is None:
            fa = eval_f(a)
            if fa is None:
                return best_x
        
        if abs(fa) <= 1e-8:
            return a
        
        b = a + 1.0
        for _ in range(30):
            if time.time() > deadline or call_count >= max_calls - 5:
                break
            fb = eval_f(b)
            if fb is None:
                break
            if not math.isfinite(fb):
                b = b * 2.0
                continue
            if abs(fb) <= 1e-8:
                return b
            if fa * fb < 0:
                # Found bracket [a, b], use bisection
                for _ in range(50):
                    if time.time() > deadline or call_count >= max_calls:
                        break
                    m = (a + b) * 0.5
                    if m == a or m == b:
                        break
                    fm = eval_f(m)
                    if fm is None:
                        break
                    if abs(fm) <= 1e-8:
                        return m
                    if fa * fm < 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                return (a + b) * 0.5
            # Expand
            a = b
            fa = fb
            b = b * 2.0 if b > 0 else b * 2.0 - 1.0
            if abs(b) > 1e6:
                break
    
    # Return best found
    if best_fx is not None and math.isfinite(best_fx):
        return best_x
    return x
# EVOLVE-BLOCK-END
