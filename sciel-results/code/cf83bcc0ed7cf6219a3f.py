import math
import time
import scipy.optimize

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    max_calls = 480
    call_count = 0
    
    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    # Helper to call f and track count
    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    fx = f_call(x)
    
    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    # --- Phase 1: Find a Bracket ---
    a = x
    fa = fx
    found = False
    b = None
    fb = None
    
    # Try small steps first, then expand. Only evaluate one side per iteration
    # to minimize calls. Alternate direction based on which side had smaller |f|.
    d = 0.5
    try_right_first = True
    
    for _ in range(35):
        if not time_ok():
            break
            
        if try_right_first:
            xr = a + d
            fr = f_call(xr)
            if math.isfinite(fr):
                if fa * fr < 0:
                    b = xr
                    fb = fr
                    found = True
                    break
                # Remember this value for potential use later
                last_right = (xr, fr)
        else:
            xl = a - d
            fl = f_call(xl)
            if math.isfinite(fl):
                if fa * fl < 0:
                    b = xl
                    fb = fl
                    found = True
                    break
                last_left = (xl, fl)
        
        # Alternate direction, but also try the other side if we haven't already
        # at this d value. Actually, let's just alternate and double d.
        try_right_first = not try_right_first
        d *= 2.0
        
        if d > 1e8:
            break

    if not found:
        # Try a few more large jumps
        for d_try in [1e4, 1e6, 1e8]:
            if not time_ok():
                break
            xr = a + d_try
            fr = f_call(xr)
            if math.isfinite(fr) and fa * fr < 0:
                b = xr
                fb = fr
                found = True
                break
            if not found:
                xl = a - d_try
                fl = f_call(xl)
                if math.isfinite(fl) and fa * fl < 0:
                    b = xl
                    fb = fl
                    found = True
                    break

    if not found:
        # Fallback: Secant method without bracket
        x1 = x0 + 1e-6
        fx1 = f_call(x1)
        
        for _ in range(30):
            if not time_ok(): 
                break
            if abs(fx) <= 1e-8: 
                return x0
            if abs(fx1) <= 1e-8: 
                return x1
            if fx1 == fx: 
                break
            
            x_new = x0 - fx * (x0 - x1) / (fx - fx1)
            if not math.isfinite(x_new): 
                break
                
            f_new = f_call(x_new)
            if not math.isfinite(f_new): 
                break
                
            x0, fx = x1, fx1
            x1, fx1 = x_new, f_new
            
        return x0 if abs(fx) < abs(fx1) else x1

    # --- Phase 2: Refine using SciPy Brent's Method ---
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    try:
        root = scipy.optimize.brentq(f_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
        
        if math.isfinite(root):
            return root
        else:
            return a

    except Exception:
        # Fallback to manual Illinois/Secant-Bisection if scipy fails
        x_prev = a
        fx_prev = fa
        x_curr = b
        fx_curr = fb
        
        for _ in range(100):
            if not time_ok():
                break
            if abs(fx_curr) <= 1e-8:
                return x_curr
            if abs(fx_prev) <= 1e-8:
                return x_prev
                
            if abs(x_curr - x_prev) < 1e-12 * max(1.0, abs(x_curr), abs(x_prev)):
                return 0.5 * (x_curr + x_prev)
                
            if fx_curr != fx_prev:
                x_secant = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            else:
                x_secant = 0.5 * (x_curr + x_prev)
                
            if min(x_prev, x_curr) < x_secant < max(x_prev, x_curr):
                x_next = x_secant
            else:
                x_next = 0.5 * (x_curr + x_prev)
                
            f_next = f_call(x_next)
            
            if not math.isfinite(f_next):
                 x_next = 0.5 * (x_curr + x_prev)
                 f_next = f_call(x_next)
                 
            if abs(f_next) <= 1e-8:
                return x_next
                 
            if fx_prev * f_next < 0:
                x_curr = x_next
                fx_curr = f_next
            else:
                x_prev = x_next
                fx_prev = f_next
                
        return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
