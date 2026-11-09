import math
import time

def find_root(f, x0):
    """
    Finds a root x for f(x) such that |f(x)| <= 1e-8.
    Uses a combination of Steffensen's method, Secant, and Bisection.
    """
    tol = 1e-8
    max_calls = 480  # Safety margin below 500
    call_count = 0
    cache = {}
    
    # Deadline guard: 110 seconds
    start_time = time.time()
    
    def safe_f(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("Call limit exceeded")
        if time.time() - start_time > 110:
            raise RuntimeError("Time limit exceeded")
            
        if x in cache:
            return cache[x]
        
        fx = f(x)
        call_count += 1
        cache[x] = fx
        
        if not math.isfinite(fx):
            raise ValueError("Non-finite value")
            
        return fx

    x = float(x0)
    
    # 1. Check initial point
    try:
        fx = safe_f(x)
    except Exception:
        return x
        
    if abs(fx) <= tol:
        return x

    best_x = x
    best_f = fx

    # 2. Try Steffensen's Method from x0
    # Steffensen: x_{n+1} = x_n - f(x_n)^2 / (f(x_n + f(x_n)) - f(x_n))
    # This is equivalent to Newton's method but uses finite differences for the derivative.
    # It has quadratic convergence near the root.
    
    x_curr = x
    f_curr = fx
    
    # Adaptive step for the finite difference in the denominator
    # We use f(x) itself as the step size, which is standard Steffensen.
    # However, if f(x) is very small or very large, this might be unstable.
    # Let's try standard Steffensen first.
    
    for i in range(50):
        if call_count > max_calls - 10:
            break
            
        # Calculate x_probe = x_curr + f_curr
        x_probe = x_curr + f_curr
        
        try:
            f_probe = safe_f(x_probe)
        except Exception:
            break
            
        if abs(f_probe) <= tol:
            return x_probe
            
        # Update best
        if abs(f_probe) < abs(best_f):
            best_f = f_probe
            best_x = x_probe
            
        denom = f_probe - f_curr
        if abs(denom) < 1e-15:
            break
            
        x_next = x_curr - (f_curr * f_curr) / denom
        
        if not math.isfinite(x_next):
            break
            
        # Safety check: if we jump too far, stop
        # For oscillatory functions, jumping too far is bad.
        # Let's limit the step size relative to current x or 1.0
        max_step = 10.0 * max(1.0, abs(x_curr))
        if abs(x_next - x_curr) > max_step:
            break
            
        try:
            f_next = safe_f(x_next)
        except Exception:
            break
            
        if abs(f_next) <= tol:
            return x_next
            
        if abs(f_next) < abs(best_f):
            best_f = f_next
            best_x = x_next
            
        # Check for convergence
        if abs(x_next - x_curr) < 1e-12 * max(1.0, abs(x_next)):
            if abs(f_next) <= 1e-7:
                # Polish with a few more Steffensen steps or just return
                # Let's do a quick polish
                for j in range(10):
                    if call_count > max_calls - 2:
                        break
                    x_p = x_next + f_next
                    try:
                        f_p = safe_f(x_p)
                        if abs(f_p) <= tol:
                            return x_p
                        d_p = f_p - f_next
                        if abs(d_p) < 1e-15:
                            break
                        x_new_p = x_next - (f_next * f_next) / d_p
                        if not math.isfinite(x_new_p):
                            break
                        f_new_p = safe_f(x_new_p)
                        if abs(f_new_p) <= tol:
                            return x_new_p
                        x_next, f_next = x_new_p, f_new_p
                    except Exception:
                        break
                return x_next
            break
            
        x_curr = x_next
        f_curr = f_next

    # 3. If Steffensen failed, find a bracket
    # We reuse the points we've visited if possible, or search outwards.
    # Let's try to find a sign change using the points we already have in cache,
    # or by searching outwards from x0.
    
    a = x
    fa = fx
    b = None
    fb = None
    found_bracket = False
    
    # Check cache for sign change with x0
    # This is expensive to iterate over, so let's just do a quick outward search.
    steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    directions = [1, -1]
    
    for step in steps:
        for d in directions:
            x_new = x + d * step
            try:
                fx_new = safe_f(x_new)
            except Exception:
                continue
                
            if abs(fx_new) <= tol:
                return x_new
                
            if abs(fx_new) < abs(best_f):
                best_f = fx_new
                best_x = x_new
                
            if fa * fx_new < 0:
                a, b = x, x_new
                fa, fb = fx, fx_new
                found_bracket = True
                break
        if found_bracket:
            break
        if call_count > max_calls - 20:
            break

    if not found_bracket:
        return best_x

    # 4. Refine with Secant, falling back to Bisection
    lo = min(a, b)
    hi = max(a, b)
    
    x_prev = a
    f_prev = fa
    x_curr = b
    f_curr = fb
    
    for i in range(40):
        if call_count > max_calls - 5:
            break
            
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            break
            
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_next):
            break
            
        # Check if x_next is in or near bracket
        # Allow slight expansion
        if x_next < lo - (hi - lo) or x_next > hi + (hi - lo):
            break
            
        try:
            f_next = safe_f(x_next)
        except Exception:
            break
            
        if abs(f_next) <= tol:
            return x_next
            
        if abs(f_next) < abs(best_f):
            best_f = f_next
            best_x = x_next
            
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_next
        f_curr = f_next
        
        if abs(x_next - x_prev) < 1e-12 * max(1.0, abs(x_next)):
            if abs(f_next) <= 1e-7:
                break

    # 5. Bisection Fallback
    a_bis = min(a, b)
    b_bis = max(a, b)
    fa_bis = fa if a_bis == a else fb
    fb_bis = fb if b_bis == b else fa
    
    # Re-evaluate if needed (should be in cache)
    if a_bis not in cache:
        fa_bis = safe_f(a_bis)
    if b_bis not in cache:
        fb_bis = safe_f(b_bis)
        
    if fa_bis * fb_bis > 0:
        return best_x

    for i in range(60):
        if call_count > max_calls - 2:
            break
            
        mid = (a_bis + b_bis) / 2.0
        f_mid = safe_f(mid)
        
        if abs(f_mid) <= tol:
            return mid
            
        if abs(f_mid) < abs(best_f):
            best_f = f_mid
            best_x = mid
            
        if fa_bis * f_mid < 0:
            b_bis = mid
            fb_bis = f_mid
        else:
            a_bis = mid
            fa_bis = f_mid
            
        if abs(b_bis - a_bis) < 1e-12:
            break
            
    # Return best_x or midpoint
    mid_final = (a_bis + b_bis) / 2.0
    if mid_final in cache:
        if abs(cache[mid_final]) < abs(best_f):
            return mid_final
    return best_x
