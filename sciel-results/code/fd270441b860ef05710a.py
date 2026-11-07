"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x = float(x0)
    deadline = time.monotonic() + 118.0
    best_x = x
    best_fx = None
    try:
        best_fx = f(x)
    except Exception:
        pass
    
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return x
    
    # Try Newton's method first
    x_newton = x
    fx = best_fx
    for _ in range(30):
        if time.monotonic() > deadline:
            break
        if fx is not None and abs(fx) <= 1e-10:
            return x_newton
        h = 1e-7 * max(1.0, abs(x_newton))
        try:
            fx_plus = f(x_newton + h)
        except Exception:
            break
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_newton = x_newton - fx / d
        if not math.isfinite(x_newton):
            break
        try:
            fx = f(x_newton)
        except Exception:
            break
        if fx is not None and abs(fx) < abs(best_fx) if best_fx is not None else True:
            best_x = x_newton
            best_fx = fx
        if fx is not None and abs(fx) <= 1e-8:
            return x_newton
    
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x
    
    # Bisection fallback with adaptive bracketing
    # Try to find a sign change around x0
    scale = max(1.0, abs(x0))
    left = x0 - scale
    right = x0 + scale
    
    try:
        f_left = f(left)
        f_right = f(right)
    except Exception:
        f_left = None
        f_right = None
    
    # If no sign change, try expanding
    if f_left is None or f_right is None or f_left * f_right > 0:
        for expand_factor in [2, 4, 8, 16, 32]:
            if time.monotonic() > deadline:
                break
            left = x0 - scale * expand_factor
            right = x0 + scale * expand_factor
            try:
                f_left = f(left)
                f_right = f(right)
            except Exception:
                continue
            if f_left is not None and f_right is not None and f_left * f_right <= 0:
                break
    
    # Bisection
    if f_left is not None and f_right is not None and f_left * f_right <= 0:
        for _ in range(60):
            if time.monotonic() > deadline:
                break
            mid = (left + right) / 2.0
            try:
                f_mid = f(mid)
            except Exception:
                break
            if f_mid is None or not math.isfinite(f_mid):
                break
            if abs(f_mid) <= 1e-10:
                return mid
            if f_mid * f_left <= 0:
                right = mid
                f_right = f_mid
            else:
                left = mid
                f_left = f_mid
            if abs(f_mid) < abs(best_fx) if best_fx is not None else True:
                best_x = mid
                best_fx = f_mid
            if f_mid is not None and abs(f_mid) <= 1e-8:
                return mid
    
    # Return best found
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x
    return best_x
# EVOLVE-BLOCK-END
