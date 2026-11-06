"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find root of f near x0 using a robust hybrid approach.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def try_newton(f, x0, max_iter=100, tol=1e-12):
        """Newton's method with numerical derivative and damping."""
        x = float(x0)
        for _ in range(max_iter):
            if time_left() < 1.0:
                break
            fx = call_f(x)
            if abs(fx) <= tol:
                return x, eval_count
            h = 1e-7 * max(1.0, abs(x))
            d = (call_f(x + h) - fx) / h
            if d == 0:
                break
            step = fx / d
            # Damping if step is too large
            if abs(step) > 10.0:
                step = 10.0 * math.copysign(1.0, step)
            x_new = x - step
            if not math.isfinite(x_new):
                break
            x = x_new
        return x, eval_count
    
    def try_secant(f, x0, max_iter=100, tol=1e-12):
        """Secant method."""
        h = 1e-4 * max(1.0, abs(x0))
        x1 = float(x0)
        x2 = x0 + h
        for _ in range(max_iter):
            if time_left() < 1.0:
                break
            f1 = call_f(x1)
            f2 = call_f(x2)
            if abs(f1) <= tol:
                return x1, eval_count
            if abs(f2) <= tol:
                return x2, eval_count
            if f1 == f2:
                break
            x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x_new):
                break
            x1 = x2
            x2 = x_new
        return x2, eval_count
    
    def try_brent_simple(f, a, b, max_iter=100, tol=1e-12):
        """Simpler robust bracketing method combining bisection and secant."""
        fa = call_f(a)
        fb = call_f(b)
        if fa * fb > 0:
            return None, eval_count
        
        c = a
        fc = fa
        d = b - a
        e = d
        mflag = True
        
        for _ in range(max_iter):
            if time_left() < 1.0:
                break
            
            if abs(fb) <= tol:
                return b, eval_count
            if fc * fb > 0:
                c = a
                fc = fa
                d = b - c
                e = d
            if abs(e) > tol:
                if mflag:
                    s = fc / fa
                    p = 2.0 * d * s
                    q = s - 1.0
                    if abs(p) >= 0.4 * abs(e * q) or abs(p) > abs(d):
                        s = d / 2.0
                        mflag = True
                    else:
                        s = p / (2.0 * q)
                        mflag = False
            else:
                s = d / 2.0
                mflag = True
            
            if abs(s) < tol:
                s = tol if s >= 0 else -tol
            elif s > 0 and s > d or s < 0 and s < -d:
                s = d / 2.0
            
            a = b
            fa = fb
            e = d
            b = a + s
            d = b - a
            fb = call_f(b)
            
            if abs(b - a) < 1e-15 * max(1.0, abs(b)):
                break
        
        return b, eval_count
    
    # Strategy: Try multiple methods and keep the best result
    x0 = float(x0)
    best_x = x0
    best_fx = None
    
    try:
        fx0 = call_f(x0)
        if abs(fx0) <= 1e-8:
            return x0
        best_x = x0
        best_fx = fx0
    except:
        pass
    
    # Try Newton
    try:
        x_newton, _ = try_newton(f, x0, max_iter=80)
        if x_newton is not None and math.isfinite(x_newton):
            fxn = call_f(x_newton)
            if abs(fxn) < abs(best_fx) if best_fx is not None else True:
                best_x = x_newton
                best_fx = fxn
            if abs(fxn) <= 1e-8:
                return x_newton
    except:
        pass
    
    if time_left() < 10:
        return best_x
    
    # Try secant
    try:
        x_secant, _ = try_secant(f, x0, max_iter=80)
        if x_secant is not None and math.isfinite(x_secant):
            fxn = call_f(x_secant)
            if abs(fxn) < abs(best_fx) if best_fx is not None else True:
                best_x = x_secant
                best_fx = fxn
            if abs(fxn) <= 1e-8:
                return x_secant
    except:
        pass
    
    if time_left() < 5:
        return best_x
    
    # Try bracketing with bisection from various initial intervals
    for step in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        if time_left() < 5 or eval_count >= MAX_EVALS:
            break
        
        for sign in [-1, 1]:
            if time_left() < 5 or eval_count >= MAX_EVALS:
                break
            a = x0
            b = x0 + sign * step
            try:
                x_bi, _ = try_brent_simple(f, a, b, max_iter=60)
                if x_bi is not None and math.isfinite(x_bi):
                    fxn = call_f(x_bi)
                    if abs(fxn) < abs(best_fx) if best_fx is not None else True:
                        best_x = x_bi
                        best_fx = fxn
                    if abs(fxn) <= 1e-8:
                        return x_bi
            except:
                pass
    
    # Try a few more Newton iterations from best_x
    if eval_count < MAX_EVALS - 10 and time_left() > 2:
        try:
            x_final, _ = try_newton(f, best_x, max_iter=30)
            if x_final is not None and math.isfinite(x_final):
                fxn = call_f(x_final)
                if abs(fxn) < abs(best_fx) if best_fx is not None else True:
                    best_x = x_final
                    best_fx = fxn
                if abs(fxn) <= 1e-8:
                    return x_final
        except:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
