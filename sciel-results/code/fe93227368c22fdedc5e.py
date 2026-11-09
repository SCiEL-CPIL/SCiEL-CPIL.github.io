import math

def find_root(f, x0):
    tol = 1e-8
    x = float(x0)
    fx = f(x)
    calls = 1

    if abs(fx) <= tol:
        return x

    x_best = x
    fx_best = fx

    # Phase 1: Newton's Method with Central Difference Derivative
    h = 1e-5
    for _ in range(10):
        x_plus_h = x + h
        x_minus_h = x - h
        f_plus_h = f(x_plus_h)
        f_minus_h = f(x_minus_h)
        calls += 2
        
        if not (math.isfinite(f_plus_h) and math.isfinite(f_minus_h)):
            break
            
        df = (f_plus_h - f_minus_h) / (2 * h)
        
        if abs(df) < 1e-15:
            break
            
        x_new = x - fx / df
        
        if not math.isfinite(x_new):
            break
            
        if abs(x_new - x) > 1000.0 * (1.0 + abs(x)):
            break
            
        f_new = f(x_new)
        calls += 1
        
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= tol:
            return x_new
            
        if abs(f_new) < abs(fx_best):
            x_best = x_new
            fx_best = f_new
            
        if abs(f_new) > abs(fx) * 1.1:
            break
            
        x = x_new
        fx = f_new

    # Phase 2: Bracketing
    a = x_best
    fa = fx_best
    b = None
    fb = None
    
    deltas = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]
    
    for delta in deltas:
        x_plus = a + delta
        f_plus = f(x_plus)
        calls += 1
        if abs(f_plus) <= tol:
            return x_plus
        if fa * f_plus < 0:
            b = x_plus
            fb = f_plus
            break
        
        x_minus = a - delta
        f_minus = f(x_minus)
        calls += 1
        if abs(f_minus) <= tol:
            return x_minus
        if fa * f_minus < 0:
            b = x_minus
            fb = f_minus
            break

    if b is None:
        return x_best

    # Phase 3: Robust Secant/Bisection
    x_a = a
    x_b = b
    f_a = fa
    f_b = fb
    
    for _ in range(400):
        if abs(f_a) <= tol:
            return x_a
        if abs(f_b) <= tol:
            return x_b
            
        if f_a == f_b:
            x_next = (x_a + x_b) / 2.0
        else:
            x_next = x_b - f_b * (x_b - x_a) / (f_b - f_a)
            
        if (x_next < min(x_a, x_b)) or (x_next > max(x_a, x_b)):
            x_next = (x_a + x_b) / 2.0
            
        f_next = f(x_next)
        calls += 1
        
        if abs(f_next) <= tol:
            return x_next
            
        if f_a * f_next < 0:
            x_b = x_next
            f_b = f_next
        else:
            x_a = x_next
            f_a = f_next
            
        if abs(x_b - x_a) < 1e-12:
            return x_a if abs(f_a) < abs(f_b) else x_b

    return x_a if abs(f_a) < abs(f_b) else x_b
