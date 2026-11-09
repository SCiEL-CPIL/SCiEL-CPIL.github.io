import time
import math
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    start_time = time.time()
    MAX_TIME = 110.0
    MAX_CALLS = 490
    calls = [0]
    cache = {}
    
    def feval(x):
        if calls[0] >= MAX_CALLS:
            return None
        if time.time() - start_time > MAX_TIME:
            return None
        if x in cache:
            return cache[x]
        calls[0] += 1
        try:
            val = f(x)
        except:
            return None
        if not math.isfinite(val):
            return None
        cache[x] = val
        return val

    x = float(x0)
    fx = feval(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_f = abs(fx)

    # Bracketing phase
    a = None
    b = None
    step = 1e-2 * max(1.0, abs(x))
    
    # We need to check intervals [x-step, x] and [x, x+step]
    # and expand.
    last_left_x = x
    last_left_f = fx
    last_right_x = x
    last_right_f = fx
    
    for i in range(40):
        if time.time() - start_time > MAX_TIME - 5:
            break
            
        new_left_x = x - step
        new_right_x = x + step
        
        new_left_f = feval(new_left_x)
        new_right_f = feval(new_right_x)
        
        if new_left_f is None or new_right_f is None:
            break
            
        if abs(new_left_f) < best_f:
            best_f = abs(new_left_f)
            best_x = new_left_x
        if abs(new_right_f) < best_f:
            best_f = abs(new_right_f)
            best_x = new_right_x
            
        # Check sign change in [new_left_x, last_left_x]
        if new_left_f * last_left_f < 0:
            a, b = new_left_x, last_left_x
            break
        # Check sign change in [last_right_x, new_right_x]
        if last_right_f * new_right_f < 0:
            a, b = last_right_x, new_right_x
            break
            
        # If no sign change, expand
        last_left_x = new_left_x
        last_left_f = new_left_f
        last_right_x = new_right_x
        last_right_f = new_right_f
        step *= 2.0

    if a is not None and b is not None:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if math.isfinite(root):
                return root
        except:
            pass

    # Fallback: minimize f(x)^2 in a local range
    # Range based on expansion
    lower = x - step
    upper = x + step
    
    try:
        res = minimize_scalar(lambda x: f(x)**2, bounds=(lower, upper), method='bounded', options={'xatol': 1e-10})
        if res.success and math.isfinite(res.x):
            x_opt = res.x
            f_opt = feval(x_opt)
            if f_opt is not None and abs(f_opt) < abs(best_f):
                best_x = x_opt
                best_f = abs(f_opt)
    except:
        pass

    return best_x
