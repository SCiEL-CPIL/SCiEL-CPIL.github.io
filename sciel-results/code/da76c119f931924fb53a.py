"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Uses a multi-phase approach:
    1. Quick check at x0
    2. Adaptive secant with bracketing
    3. Newton with finite-difference derivative
    4. Final polish
    """
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    def time_ok():
        return time.time() < deadline
    
    x = float(x0)
    if not math.isfinite(x):
        return x0
    
    # Count calls carefully
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    
    if not math.isfinite(best_fx):
        return x0
    
    if best_abs <= 1e-8:
        return x
    
    # Phase 1: Adaptive sampling to find a bracket or get close
    # Try multiple step sizes to find a sign change or small value
    h_base = max(1e-6, 1e-4 * abs(x))
    candidates = []
    
    for scale in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
        h = scale * max(1.0, abs(x))
        for direction in [1, -1]:
            if not time_ok():
                break
            x_try = x + direction * h
            if not math.isfinite(x_try):
                continue
            fx_try = f(x_try)
            if not math.isfinite(fx_try):
                continue
            candidates.append((x_try, fx_try))
            if abs(fx_try) < best_abs:
                best_x = x_try
                best_fx = fx_try
                best_abs = abs(fx_try)
            if best_abs <= 1e-8:
                return best_x
    
    # Check for sign changes among candidates to find a bracket
    # Sort candidates by x value
    candidates.sort(key=lambda p: p[0])
    bracket = None
    for i in range(len(candidates) - 1):
        x1, f1 = candidates[i]
        x2, f2 = candidates[i + 1]
        if f1 * f2 < 0:
            bracket = (x1, f1, x2, f2)
            break
    
    if bracket:
        # Use bisection within the bracket
        a, fa, b, fb = bracket
        for _ in range(60):
            if not time_ok():
                break
            m = 0.5 * (a + b)
            fm = f(m)
            if not math.isfinite(fm):
                break
            if abs(fm) < best_abs:
                best_x = m
                best_fx = fm
                best_abs = abs(fm)
            if best_abs <= 1e-8:
                return best_x
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
            if abs(b - a) < 1e-12 * max(1.0, abs(a)):
                break
        # Return midpoint
        mid = 0.5 * (a + b)
        fm = f(mid)
        if math.isfinite(fm) and abs(fm) < best_abs:
            best_x = mid
            best_fx = fm
            best_abs = abs(fm)
        if best_abs <= 1e-8:
            return best_x
        return best_x
    
    # Phase 2: Secant method from best point
    if len(candidates) >= 1:
        # Pick the two closest candidates to best_x
        candidates.sort(key=lambda p: abs(p[0] - best_x))
        if len(candidates) >= 2:
            x_curr = candidates[0][0]
            f_curr = candidates[0][1]
            x_prev = candidates[1][0]
            f_prev = candidates[1][1]
        else:
            x_curr = best_x
            f_curr = best_fx
            x_prev = best_x + 1e-4 * max(1.0, abs(best_x))
            f_prev = f(x_prev)
            if not math.isfinite(f_prev):
                f_prev = f(best_x - 1e-4 * max(1.0, abs(best_x)))
                x_prev = best_x - 1e-4 * max(1.0, abs(best_x))
                if not math.isfinite(f_prev):
                    return best_x if best_abs <= 1e-8 else x0
    
        for i in range(20):
            if not time_ok():
                break
            if abs(f_curr) <= 1e-8:
                return x_curr
            if abs(f_curr) < best_abs:
                best_x = x_curr
                best_fx = f_curr
                best_abs = abs(f_curr)
            
            if abs(f_curr - f_prev) < 1e-30:
                break
            
            x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            if not math.isfinite(x_next):
                break
            
            # Damping if step is too large
            step = abs(x_next - x_curr)
            max_step = 10.0 * max(1.0, abs(x_curr))
            if step > max_step:
                x_next = x_curr + math.copysign(max_step, x_next - x_curr)
            
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f(x_next)
            
            if not math.isfinite(f_curr):
                x_curr, f_curr = x_prev, f_prev
                break
    
    if abs(f_curr) <= 1e-8:
        return x_curr
    if abs(f_curr) < best_abs:
        best_x = x_curr
        best_fx = f_curr
        best_abs = abs(f_curr)
    
    # Phase 3: Newton's method with finite-difference derivative
    x = best_x
    fx = best_fx
    
    for i in range(30):
        if not time_ok():
            break
        if abs(fx) <= 1e-8:
            return x
        if abs(fx) < best_abs:
            best_x = x
            best_fx = fx
            best_abs = abs(fx)
        
        # Adaptive step for finite difference
        h = 1e-7 * max(1.0, abs(x))
        fx_h = f(x + h)
        
        if not math.isfinite(fx_h):
            h = 1e-5 * max(1.0, abs(x))
            fx_h = f(x + h)
            if not math.isfinite(fx_h):
                break
        
        d = (fx_h - fx) / h
        if abs(d) < 1e-30:
            break
        
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
        
        # Damping
        step = abs(x_new - x)
        max_step = 5.0 * max(1.0, abs(x))
        if step > max_step:
            x_new = x + math.copysign(max_step, x_new - x)
        
        x = x_new
        fx = f(x)
        
        if not math.isfinite(fx):
            break
    
    if abs(fx) <= 1e-8:
        return x
    if abs(fx) < best_abs:
        best_x = x
        best_fx = fx
        best_abs = abs(fx)
    
    # Phase 4: Final polish with tighter tolerance
    x = best_x
    fx = best_fx
    
    for i in range(15):
        if not time_ok():
            break
        if abs(fx) <= 1e-10:
            return x
        if abs(fx) < best_abs:
            best_x = x
            best_fx = fx
            best_abs = abs(fx)
        
        h = 1e-8 * max(1.0, abs(x))
        fx_h = f(x + h)
        if not math.isfinite(fx_h):
            break
        d = (fx_h - fx) / h
        if abs(d) < 1e-30:
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new
        fx = f(x)
        if not math.isfinite(fx):
            break
    
    if abs(fx) <= 1e-8:
        return x
    
    # Return best found
    if best_abs <= 1e-8:
        return best_x
    
    # Last resort
    if math.isfinite(x) and abs(fx) < best_abs:
        return x
    if math.isfinite(best_x):
        return best_x
    return x0
# EVOLVE-BLOCK-END
