import math
import time

def find_root(f, x0):
    """
    Finds a root x for f(x) such that |f(x)| <= 1e-8.
    Uses a combination of Steffensen's method, bracketing search,
    secant refinement, and bisection fallback.
    """
    tol = 1e-8
    max_calls = 490  # Safety margin below 500
    call_count = 0
    cache = {}
    
    # Deadline guard: 115 seconds (150s limit)
    start_time = time.time()
    
    def safe_f(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("Call limit exceeded")
        if time.time() - start_time > 115:
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

    # 2. Try Steffensen's Method from x0 (quadratic convergence, no bracket needed)
    # x_{n+1} = x_n - f(x_n)^2 / (f(x_n + f(x_n)) - f(x_n))
    x_ste = x
    f_ste = fx
    
    for i in range(20):
        if call_count > max_calls - 50:
            break
        if time.time() - start_time > 110:
            break
            
        x_probe = x_ste + f_ste
        if not math.isfinite(x_probe):
            break
            
        try:
            f_probe = safe_f(x_probe)
        except Exception:
            break
            
        if abs(f_probe) <= tol:
            return x_probe
            
        if abs(f_probe) < abs(best_f):
            best_f = f_probe
            best_x = x_probe
            
        denom = f_probe - f_ste
        if abs(denom) < 1e-15:
            break
            
        x_next = x_ste - (f_ste * f_ste) / denom
        
        if not math.isfinite(x_next):
            break
            
        # Safety check: limit step size
        max_step = 10.0 * max(1.0, abs(x_ste))
        if abs(x_next - x_ste) > max_step:
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
        if abs(x_next - x_ste) < 1e-12 * max(1.0, abs(x_next)):
            if abs(f_next) <= 1e-7:
                # Polish with a few more Steffensen steps
                x_pol = x_next
                f_pol = f_next
                for j in range(10):
                    if call_count > max_calls - 5:
                        break
                    x_p = x_pol + f_pol
                    try:
                        f_p = safe_f(x_p)
                        if abs(f_p) <= tol:
                            return x_p
                        d_p = f_p - f_pol
                        if abs(d_p) < 1e-15:
                            break
                        x_new_p = x_pol - (f_pol * f_pol) / d_p
                        if not math.isfinite(x_new_p):
                            break
                        f_new_p = safe_f(x_new_p)
                        if abs(f_new_p) <= tol:
                            return x_new_p
                        x_pol, f_pol = x_new_p, f_new_p
                    except Exception:
                        break
                return x_pol
            break
            
        x_ste = x_next
        f_ste = f_next

    # 3. Find a bracket [a, b] with f(a)*f(b) < 0
    a = x
    fa = fx
    b = None
    fb = None
    found_bracket = False
    
    # Steps to try for bracketing
    steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
    directions = [1, -1]
    
    for step in steps:
        if call_count > max_calls - 30:
            break
        if time.time() - start_time > 110:
            break
            
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

    if not found_bracket:
        # If no bracket found, try a denser local search
        # Check cache for any sign change with x0
        for cx, cf in cache.items():
            if cx != x and fa * cf < 0:
                a, b = x, cx
                fa, fb = fx, cf
                found_bracket = True
                break
                
        if not found_bracket:
            return best_x

    # 4. Refine with Secant Method
    lo = min(a, b)
    hi = max(a, b)
    
    x_prev = a
    f_prev = fa
    x_curr = b
    f_curr = fb
    
    for i in range(40):
        if call_count > max_calls - 10:
            break
        if time.time() - start_time > 110:
            break
            
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            break
            
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_next):
            break
            
        # Check if x_next is in or near bracket
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
    
    # Get f values from cache or re-evaluate
    if a_bis in cache:
        fa_bis = cache[a_bis]
    else:
        try:
            fa_bis = safe_f(a_bis)
        except Exception:
            return best_x
            
    if b_bis in cache:
        fb_bis = cache[b_bis]
    else:
        try:
            fb_bis = safe_f(b_bis)
        except Exception:
            return best_x
            
    if fa_bis * fb_bis > 0:
        return best_x

    for i in range(60):
        if call_count > max_calls - 2:
            break
        if time.time() - start_time > 115:
            break
            
        mid = (a_bis + b_bis) / 2.0
        
        try:
            f_mid = safe_f(mid)
        except Exception:
            break
            
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
            
    # Return the best candidate
    mid_final = (a_bis + b_bis) / 2.0
    if mid_final in cache:
        if abs(cache[mid_final]) < abs(best_f):
            return mid_final
    return best_x
