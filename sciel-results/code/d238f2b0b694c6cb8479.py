"""Root finder for 3 fixed oscillatory equations with call budget management."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 105.0  # Safety margin for 120s limit
    
    calls = 0
    MAX_CALLS = 450  # Stay well under 500
    
    best_x = float(x0)
    best_fx = float('inf')
    
    def _f(x):
        nonlocal calls, best_x, best_fx
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError()
        if calls >= MAX_CALLS:
            raise MemoryError()
        calls += 1
        try:
            val = float(f(x))
        except Exception:
            val = float('inf')
        if math.isfinite(val) and abs(val) < abs(best_fx):
            best_fx = val
            best_x = x
        return val
    
    def _time_ok():
        return time.time() - start_time < TIME_LIMIT - 2.0
    
    def _calls_ok():
        return calls < MAX_CALLS - 5
    
    def bisection(a, b, fa, fb, max_iter=60):
        """Bisection method with call tracking."""
        nonlocal calls, best_x, best_fx
        for _ in range(max_iter):
            if not _time_ok() or not _calls_ok():
                break
            mid = 0.5 * (a + b)
            if mid == a or mid == b:
                break
            fm = _f(mid)
            if not math.isfinite(fm):
                break
            if abs(fm) < abs(best_fx):
                best_fx = fm
                best_x = mid
            if abs(fm) <= 1e-12:
                return mid
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                break
        return best_x
    
    def illinois(a, b, fa, fb, max_iter=60):
        """Illinois method (modified bisection with exponential weighting)."""
        nonlocal calls, best_x, best_fx
        for _ in range(max_iter):
            if not _time_ok() or not _calls_ok():
                break
            # Illinois: weight based on previous interval sizes
            if fa * fb > 0:
                break
            # Compute Illinois point
            if abs(fa) < abs(fb):
                c = (a * fb - b * fa) / (fb - fa)
            else:
                c = 0.5 * (a + b)
            # Ensure c is strictly between a and b
            if c <= a or c >= b:
                c = 0.5 * (a + b)
            fc = _f(c)
            if not math.isfinite(fc):
                break
            if abs(fc) < abs(best_fx):
                best_fx = fc
                best_x = c
            if abs(fc) <= 1e-12:
                return c
            if fa * fc <= 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            if abs(b - a) < 1e-14 * max(1.0, abs(c)):
                break
        return best_x
    
    def newton_refine(x, fx, max_iter=20):
        """Newton's method refinement with fallback to bisection."""
        nonlocal calls, best_x, best_fx
        x = float(x)
        fx = float(fx)
        for _ in range(max_iter):
            if not _time_ok() or not _calls_ok():
                break
            if abs(fx) <= 1e-12:
                return x
            h = 1e-8 * max(1.0, abs(x))
            fx_plus = _f(x + h)
            if not math.isfinite(fx_plus):
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            step = fx / d
            x_new = x - step
            if not math.isfinite(x_new):
                break
            # Limit step size
            if abs(step) > 10.0 * max(1.0, abs(x)):
                x_new = x + math.copysign(min(abs(step), 10.0 * max(1.0, abs(x))), step)
            fx_new = _f(x_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) < abs(best_fx):
                best_fx = fx_new
                best_x = x_new
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                # Step didn't help, try smaller step
                x_new = x - 0.5 * step
                fx_new = _f(x_new)
                if not math.isfinite(fx_new):
                    break
                if abs(fx_new) < abs(best_fx):
                    best_fx = fx_new
                    best_x = x_new
                if abs(fx_new) < abs(fx):
                    x = x_new
                    fx = fx_new
                else:
                    break
        return best_x
    
    def find_bracket(x0, direction, max_steps=200):
        """
        Walk from x0 in a given direction to find a sign change.
        Uses intermediate midpoint checks to avoid jumping over roots.
        """
        nonlocal calls, best_x, best_fx
        x = float(x0)
        fx = _f(x)
        if not math.isfinite(fx):
            return None
        if abs(fx) <= 1e-12:
            return (x, x, fx, fx)
        
        step_size = 0.1 * max(1.0, abs(x))
        prev_x = x
        prev_fx = fx
        
        for i in range(max_steps):
            if not _time_ok() or not _calls_ok():
                break
            next_x = x + direction * step_size
            next_fx = _f(next_x)
            if not math.isfinite(next_fx):
                # Try smaller step
                step_size *= 0.5
                continue
            
            # Check for sign change between prev and next
            if prev_fx * next_fx <= 0:
                # Found bracket, but check midpoint to avoid missing fast oscillations
                mid = 0.5 * (prev_x + next_x)
                if _calls_ok() and _time_ok():
                    mid_fx = _f(mid)
                    if math.isfinite(mid_fx):
                        if abs(mid_fx) < abs(best_fx):
                            best_fx = mid_fx
                            best_x = mid
                        if abs(mid_fx) <= 1e-12:
                            return (mid, mid, mid_fx, mid_fx)
                        if prev_fx * mid_fx <= 0:
                            return (prev_x, mid, prev_fx, mid_fx)
                        elif mid_fx * next_fx <= 0:
                            return (mid, next_x, mid_fx, next_fx)
                return (prev_x, next_x, prev_fx, next_fx)
            
            # Update
            prev_x = x
            prev_fx = fx
            x = next_x
            fx = next_fx
            
            # Adaptive step size: grow if no sign change, shrink if getting large
            if i < 10:
                step_size *= 1.5
            else:
                step_size *= 1.2
            step_size = min(step_size, 100.0 * max(1.0, abs(x0)))
        
        return None
    
    # Strategy: Try Newton first, then bracketing if that fails
    x0 = float(x0)
    
    # Phase 1: Try Newton's method from x0
    try:
        fx0 = _f(x0)
        if math.isfinite(fx0) and abs(fx0) <= 1e-12:
            return x0
        if math.isfinite(fx0):
            newton_result = newton_refine(x0, fx0, max_iter=15)
            if math.isfinite(newton_result):
                check_fx = _f(newton_result)
                if math.isfinite(check_fx) and abs(check_fx) <= 1e-8:
                    return newton_result
    except (TimeoutError, MemoryError):
        pass
    except Exception:
        pass
    
    # Phase 2: Bracketing search
    # Try both directions from x0
    for direction in [1, -1]:
        if not _time_ok() or not _calls_ok():
            break
        bracket = find_bracket(x0, direction, max_steps=100)
        if bracket is not None:
            a, b, fa, fb = bracket
            if not _time_ok() or not _calls_ok():
                break
            # Use Illinois method for robustness
            result = illinois(a, b, fa, fb, max_iter=50)
            if math.isfinite(result):
                check_fx = _f(result)
                if math.isfinite(check_fx) and abs(check_fx) <= 1e-8:
                    return result
            # Also try bisection as fallback
            if _time_ok() and _calls_ok():
                result_b = bisection(a, b, fa, fb, max_iter=50)
                if math.isfinite(result_b):
                    check_fx = _f(result_b)
                    if math.isfinite(check_fx) and abs(check_fx) <= 1e-8:
                        return result_b
    
    # Phase 3: Broader search with coarser grid if previous failed
    if _time_ok() and _calls_ok():
        # Sample a wider range
        base = abs(x0) + 1.0
        for scale in [1, 5, 20, 100]:
            if not _time_ok() or not _calls_ok():
                break
            search_radius = base * scale
            n_points = 30
            for i in range(n_points + 1):
                if not _time_ok() or not _calls_ok():
                    break
                x_sample = x0 - search_radius + 2.0 * search_radius * i / n_points
                try:
                    fx_sample = _f(x_sample)
                    if not math.isfinite(fx_sample):
                        continue
                    if abs(fx_sample) <= 1e-8:
                        return x_sample
                except (TimeoutError, MemoryError):
                    break
                except Exception:
                    continue
            
            # Try to find brackets in this range
            if _time_ok() and _calls_ok():
                prev_x = None
                prev_fx = None
                for i in range(n_points + 1):
                    if not _time_ok() or not _calls_ok():
                        break
                    x_sample = x0 - search_radius + 2.0 * search_radius * i / n_points
                    try:
                        fx_sample = _f(x_sample)
                        if not math.isfinite(fx_sample):
                            prev_x = None
                            continue
                    except (TimeoutError, MemoryError):
                        break
                    except Exception:
                        prev_x = None
                        continue
                    
                    if prev_x is not None and prev_fx is not None and prev_fx * fx_sample <= 0:
                        # Found bracket
                        if _time_ok() and _calls_ok():
                            result = illinois(prev_x, x_sample, prev_fx, fx_sample, max_iter=40)
                            if math.isfinite(result):
                                check_fx = _f(result)
                                if math.isfinite(check_fx) and abs(check_fx) <= 1e-8:
                                    return result
                    prev_x = x_sample
                    prev_fx = fx_sample
    
    # Return best result found
    if math.isfinite(best_fx) and abs(best_fx) <= 1e-8:
        return best_x
    
    # Final check on x0
    try:
        fx_final = _f(x0)
        if math.isfinite(fx_final) and abs(fx_final) <= 1e-8:
            return x0
    except (TimeoutError, MemoryError):
        pass
    except Exception:
        pass
    
    return best_x if math.isfinite(best_x) else x0

# EVOLVE-BLOCK-END
