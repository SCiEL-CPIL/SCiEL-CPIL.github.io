"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using a robust hybrid approach.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    # Track best point found
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except Exception:
        return best_x
    
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is not None and (best_fx is None or abs(fx) < abs(best_fx)):
            best_x = x
            best_fx = fx
    
    # Try multiple strategies
    # Strategy 1: Secant-first with bracket fallback (1 eval/iter primary)
    try:
        result = _secant_primary(f, x0, call_f, check_time, update_best)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Strategy 2: Bisection with wider bracket expansion
    try:
        result = _bisection_bracket(f, x0, call_f, check_time, update_best)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Strategy 3: Damped secant
    try:
        result = _damped_secant(f, x0, call_f, check_time, update_best)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Strategy 4: Newton with wider adaptive step
    try:
        result = _adaptive_newton(f, x0, call_f, check_time, update_best)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Final polish: try to tighten the best result
    if best_fx is not None and abs(best_fx) < 1e-4:
        try:
            result = _polish(f, best_x, call_f, check_time, update_best)
            if result is not None:
                return result
        except Exception:
            pass
    
    # Return best found
    return best_x


def _secant_primary(f, x0, call_f, check_time, update_best):
    """Secant-first method with bracket fallback for ill-conditioned problems."""
    x = float(x0)
    fx = call_f(x)
    update_best(x, fx)
    
    if fx is None or abs(fx) <= 1e-8:
        return x if abs(fx) <= 1e-8 else None
    
    # Try to establish a bracket (30 steps for wider search)
    a, b = None, None
    fa, fb = None, None
    
    h = max(1e-3, abs(x) * 1e-3)
    for i in range(30):
        if not check_time():
            return None
        x_plus = x + h
        try:
            fp = call_f(x_plus)
        except Exception:
            fp = None
        update_best(x_plus, fp)
        
        if fp is not None:
            if fp * fx <= 0:
                a, b = x, x_plus
                fa, fb = fx, fp
                break
        
        x_minus = x - h
        try:
            fm = call_f(x_minus)
        except Exception:
            fm = None
        update_best(x_minus, fm)
        
        if fm is not None:
            if fm * fx <= 0:
                a, b = x_minus, x
                fa, fb = fm, fx
                break
        
        h *= 2.0
    
    # Initialize secant with a small perturbation
    h_init = 1e-6 * max(1.0, abs(x))
    x_prev = x - h_init
    try:
        f_prev = call_f(x_prev)
    except Exception:
        f_prev = None
    update_best(x_prev, f_prev)
    
    if f_prev is None:
        return None
    
    for _ in range(200):
        if not check_time():
            return None
        if abs(fx) <= 1e-8:
            return x
        
        # If we have a bracket and x is outside it, use bisection
        if a is not None and b is not None and fa is not None and fb is not None:
            if (x < a or x > b) and fa * fb <= 0:
                mid = 0.5 * (a + b)
                try:
                    fmid = call_f(mid)
                except Exception:
                    break
                update_best(mid, fmid)
                if fmid is None:
                    break
                if abs(fmid) <= 1e-8:
                    return mid
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
                x_prev, x = x, mid
                f_prev, fx = fx, fmid
                continue
        
        # Secant step (primary, 1 eval per iteration)
        if f_prev is not None and fx != f_prev and abs(fx - f_prev) > 1e-15:
            x_new = x - fx * (x - x_prev) / (fx - f_prev)
        else:
            # Fallback: Newton with forward difference
            h = 1e-8 * max(1.0, abs(x))
            try:
                fxp = call_f(x + h)
            except Exception:
                break
            if fxp is None:
                break
            d = (fxp - fx) / h
            if abs(d) < 1e-15:
                break
            x_new = x - fx / d
        
        # Wider damping for ill-conditioned cases
        step = x_new - x
        if not math.isfinite(x_new):
            break
        if abs(step) > 50 * max(1.0, abs(x)):
            step = 50 * max(1.0, abs(x)) * math.copysign(1, step)
        
        x_prev, x = x, x + step
        try:
            fx = call_f(x)
        except Exception:
            break
        update_best(x, fx)
        
        if fx is None:
            break
        if abs(fx) <= 1e-8:
            return x
    
    return None


def _bisection_bracket(f, x0, call_f, check_time, update_best):
    """Bisection with wider bracket expansion from x0."""
    x = float(x0)
    try:
        fx = call_f(x)
    except Exception:
        return None
    update_best(x, fx)
    
    if fx is None:
        return None
    if abs(fx) <= 1e-8:
        return x
    
    # Find bracket with wider search
    a, b, fa, fb = None, None, None, None
    h = max(1e-3, abs(x) * 1e-3)
    for i in range(35):
        if not check_time():
            return None
        x_plus = x + h
        try:
            fp = call_f(x_plus)
        except Exception:
            fp = None
        update_best(x_plus, fp)
        
        if fp is not None and fp * fx <= 0:
            a, b, fa, fb = x, x_plus, fx, fp
            break
        
        x_minus = x - h
        try:
            fm = call_f(x_minus)
        except Exception:
            fm = None
        update_best(x_minus, fm)
        
        if fm is not None and fm * fx <= 0:
            a, b, fa, fb = x_minus, x, fm, fx
            break
        
        h *= 2.0
    
    if a is None or b is None:
        return None
    
    # Bisection with tighter convergence
    for _ in range(120):
        if not check_time():
            return None
        mid = 0.5 * (a + b)
        try:
            fmid = call_f(mid)
        except Exception:
            break
        update_best(mid, fmid)
        if fmid is None:
            break
        if abs(fmid) <= 1e-8:
            return mid
        if fa * fmid <= 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid
        if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
            break
    
    return None


def _damped_secant(f, x0, call_f, check_time, update_best):
    """Secant method with damping."""
    x = float(x0)
    try:
        fx = call_f(x)
    except Exception:
        return None
    update_best(x, fx)
    
    if fx is None:
        return None
    if abs(fx) <= 1e-8:
        return x
    
    h = 1e-6 * max(1.0, abs(x))
    x_prev = x - h
    try:
        f_prev = call_f(x_prev)
    except Exception:
        return None
    update_best(x_prev, f_prev)
    
    if f_prev is None:
        return None
    
    for _ in range(150):
        if not check_time():
            return None
        if abs(fx) <= 1e-8:
            return x
        
        if f_prev != fx and abs(fx - f_prev) > 1e-15:
            x_new = x - fx * (x - x_prev) / (fx - f_prev)
        else:
            # Fallback to Newton
            h2 = 1e-8 * max(1.0, abs(x))
            try:
                fxp = call_f(x + h2)
            except Exception:
                break
            if fxp is None:
                break
            d = (fxp - fx) / h2
            if abs(d) < 1e-15:
                break
            x_new = x - fx / d
        
        # Damping
        step = x_new - x
        if abs(step) > 5 * max(1.0, abs(x)):
            step = 5 * max(1.0, abs(x)) * math.copysign(1, step)
        
        x_prev, x = x, x + step
        f_prev, fx = fx, call_f(x)
        update_best(x, fx)
        
        if abs(fx) <= 1e-8:
            return x
    
    return None


def _adaptive_newton(f, x0, call_f, check_time, update_best):
    """Newton's method with wider adaptive step size."""
    x = float(x0)
    try:
        fx = call_f(x)
    except Exception:
        return None
    update_best(x, fx)
    
    if fx is None:
        return None
    if abs(fx) <= 1e-8:
        return x
    
    step_size = 1.0
    for _ in range(120):
        if not check_time():
            return None
        if abs(fx) <= 1e-8:
            return x
        
        # Central difference for better accuracy
        h = 1e-7 * max(1.0, abs(x))
        try:
            fxp = call_f(x + h)
            fxm = call_f(x - h)
        except Exception:
            # Try forward difference
            try:
                fxp = call_f(x + h)
                fxm = None
            except Exception:
                break
        if fxp is None:
            break
        if fxm is not None:
            d = (fxp - fxm) / (2 * h)
        else:
            d = (fxp - fx) / h
        
        if abs(d) < 1e-15:
            break
        
        x_new = x - fx / d
        
        # Wider adaptive damping for ill-conditioned cases
        if not math.isfinite(x_new) or abs(x_new - x) > 50 * max(1.0, abs(x)):
            step_size *= 0.5
            x_new = x - step_size * fx / d
            if not math.isfinite(x_new):
                break
        else:
            step_size = min(1.0, step_size * 1.5)
        
        x = x_new
        try:
            fx = call_f(x)
        except Exception:
            break
        update_best(x, fx)
        
        if fx is None:
            break
        if abs(fx) <= 1e-8:
            return x
    
    return None


def _polish(f, x_start, call_f, check_time, update_best):
    """Final polish pass with tighter tolerance."""
    x = float(x_start)
    try:
        fx = call_f(x)
    except Exception:
        return None
    update_best(x, fx)
    
    if fx is None:
        return None
    if abs(fx) <= 1e-8:
        return x
    
    # Use secant with very tight tolerance
    h = 1e-10 * max(1.0, abs(x))
    x_prev = x - h
    try:
        f_prev = call_f(x_prev)
    except Exception:
        return None
    update_best(x_prev, f_prev)
    
    if f_prev is None:
        return None
    
    for _ in range(50):
        if not check_time():
            return None
        if abs(fx) <= 1e-8:
            return x
        
        if fx != f_prev and abs(fx - f_prev) > 1e-18:
            x_new = x - fx * (x - x_prev) / (fx - f_prev)
        else:
            break
        
        if not math.isfinite(x_new):
            break
        
        step = x_new - x
        if abs(step) > 10 * max(1.0, abs(x)):
            step = 10 * max(1.0, abs(x)) * math.copysign(1, step)
        
        x_prev, x = x, x + step
        try:
            fx = call_f(x)
        except Exception:
            break
        update_best(x, fx)
        
        if fx is None:
            break
        if abs(fx) <= 1e-8:
            return x
    
    return None
# EVOLVE-BLOCK-END
