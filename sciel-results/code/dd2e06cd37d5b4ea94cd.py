"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 45.0  # 3 equations * ~45s each = 135s < 150s limit
    
    x = float(x0)
    best_x = x
    best_fx = None
    
    try:
        best_fx = f(x)
    except Exception:
        return x
    
    if not math.isfinite(best_fx):
        return x
    
    if abs(best_fx) <= 1e-8:
        return best_x
    
    # Secant state
    x_prev = None
    fx_prev = None
    
    # Bracket for bisection fallback
    bracket_a = None
    bracket_b = None
    
    # Initialize bracket if possible
    if math.isfinite(best_fx):
        # Try to establish a bracket around x0
        for scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
            xa = x - scale
            xb = x + scale
            try:
                fa = f(xa)
                fb = f(xb)
            except Exception:
                continue
            if math.isfinite(fa) and math.isfinite(fb):
                if fa * fb < 0:
                    bracket_a = xa
                    bracket_b = xb
                    break
                elif abs(fa) < abs(best_fx):
                    bracket_a = xa
                    bracket_b = x
                    best_x = xa
                    best_fx = fa
                elif abs(fb) < abs(best_fx):
                    bracket_a = x
                    bracket_b = xb
                    best_x = xb
                    best_fx = fb
        
        # Also check x0 itself as one end
        if bracket_a is None and math.isfinite(best_fx):
            # Try smaller steps
            for scale in [0.5, 0.1, 0.01, 0.001]:
                xa = x - scale
                xb = x + scale
                try:
                    fa = f(xa)
                    fb = f(xb)
                except Exception:
                    continue
                if math.isfinite(fa) and math.isfinite(fb):
                    if fa * fb < 0:
                        bracket_a = xa
                        bracket_b = xb
                        break
    
    x_current = x
    fx_current = best_fx
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Update best
        if abs(fx_current) < abs(best_fx):
            best_fx = fx_current
            best_x = x_current
        
        if abs(fx_current) <= 1e-12:
            return x_current
        
        # Try secant step if we have two points
        if x_prev is not None and fx_prev is not None and fx_prev != fx_current:
            # Secant method
            denom = fx_current - fx_prev
            if abs(denom) > 1e-15:
                x_new = x_current - fx_current * (x_current - x_prev) / denom
                if math.isfinite(x_new):
                    # Check if within bracket
                    in_bracket = True
                    if bracket_a is not None and bracket_b is not None:
                        lo = min(bracket_a, bracket_b)
                        hi = max(bracket_a, bracket_b)
                        if x_new < lo or x_new > hi:
                            in_bracket = False
                    
                    if in_bracket:
                        x_prev = x_current
                        fx_prev = fx_current
                        x_current = x_new
                        try:
                            fx_current = f(x_current)
                        except Exception:
                            break
                        if not math.isfinite(fx_current):
                            break
                        continue
        
        # Newton step with central difference
        h = 1e-5 * max(1.0, abs(x_current))
        try:
            fx_plus = f(x_current + h)
            fx_minus = f(x_current - h)
        except Exception:
            break
        
        if not (math.isfinite(fx_plus) and math.isfinite(fx_minus)):
            break
        
        dfx = (fx_plus - fx_minus) / (2.0 * h)
        
        if abs(dfx) < 1e-12:
            # Derivative too small, try bisection if bracket exists
            if bracket_a is not None and bracket_b is not None:
                mid = (bracket_a + bracket_b) / 2.0
                x_prev = x_current
                fx_prev = fx_current
                x_current = mid
                try:
                    fx_current = f(x_current)
                except Exception:
                    break
                if not math.isfinite(fx_current):
                    break
                # Update bracket
                if fx_current * best_fx < 0:
                    bracket_b = x_current
                else:
                    bracket_a = x_current
                continue
            else:
                break
        
        x_new = x_current - fx_current / dfx
        
        if not math.isfinite(x_new):
            break
        
        # Damping: limit step size
        max_step = 1.0 + abs(x_current)
        step = x_new - x_current
        if abs(step) > max_step:
            x_new = x_current + math.copysign(max_step, step)
        
        # Check if within bracket
        in_bracket = True
        if bracket_a is not None and bracket_b is not None:
            lo = min(bracket_a, bracket_b)
            hi = max(bracket_a, bracket_b)
            if x_new < lo or x_new > hi:
                in_bracket = False
        
        if in_bracket:
            x_prev = x_current
            fx_prev = fx_current
            x_current = x_new
            try:
                fx_current = f(x_current)
            except Exception:
                break
            if not math.isfinite(fx_current):
                break
        else:
            # Newton step went outside bracket, use bisection instead
            if bracket_a is not None and bracket_b is not None:
                mid = (bracket_a + bracket_b) / 2.0
                x_prev = x_current
                fx_prev = fx_current
                x_current = mid
                try:
                    fx_current = f(x_current)
                except Exception:
                    break
                if not math.isfinite(fx_current):
                    break
                # Update bracket
                if fx_current * best_fx < 0:
                    bracket_b = x_current
                else:
                    bracket_a = x_current
            else:
                # No bracket, accept Newton step but be cautious
                x_prev = x_current
                fx_prev = fx_current
                x_current = x_new
                try:
                    fx_current = f(x_current)
                except Exception:
                    break
                if not math.isfinite(fx_current):
                    break
    
    # Final polish: if we're close, do a few more Newton steps with tighter tolerance
    if abs(best_fx) <= 1e-6:
        x_polish = best_x
        fx_polish = best_fx
        for _ in range(20):
            if time.time() > deadline:
                break
            if abs(fx_polish) <= 1e-12:
                return x_polish
            
            h = 1e-6 * max(1.0, abs(x_polish))
            try:
                fxp = f(x_polish + h)
                fxm = f(x_polish - h)
            except Exception:
                break
            
            if not (math.isfinite(fxp) and math.isfinite(fxm)):
                break
            
            dfp = (fxp - fxm) / (2.0 * h)
            if abs(dfp) < 1e-14:
                break
            
            x_new = x_polish - fx_polish / dfp
            if not math.isfinite(x_new):
                break
            
            try:
                fx_new = f(x_new)
            except Exception:
                break
            
            if not math.isfinite(fx_new):
                break
            
            if abs(fx_new) < abs(fx_polish):
                x_polish = x_new
                fx_polish = fx_new
            else:
                break
        
        if abs(fx_polish) < abs(best_fx):
            best_x = x_polish
            best_fx = fx_polish
    
    return best_x
# EVOLVE-BLOCK-END
