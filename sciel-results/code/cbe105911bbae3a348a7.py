import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0
    max_calls = 480
    call_count = 0

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    if not math.isfinite(x):
        return x

    fx = f_call(x)

    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    # Step 1: Find a bracket by expanding in both directions
    # Try multiple expansion strategies and keep the best (tightest) bracket
    best_bracket = None  # (a, b, fa, fb) with smallest |a-b|
    
    def try_bracket_expand(x0_val, f0_val, direction, max_expansions=35):
        """Expand in one direction to find a sign change."""
        a = x0_val
        fa = f0_val
        b = None
        fb = None
        h = 1.0
        
        for i in range(max_expansions):
            if not time_ok():
                break
            b = a + direction * h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if fa * fb < 0:
                return (a, b, fa, fb)
            # Also check if we crossed a root (same sign but very close)
            a = b
            fa = fb
            h *= 2.0
        
        return None

    # Try expanding right
    br = try_bracket_expand(x, fx, 1.0)
    if br is not None:
        best_bracket = br
    
    # Try expanding left
    bl = try_bracket_expand(x, fx, -1.0)
    if bl is not None:
        if best_bracket is None or abs(bl[1] - bl[0]) < abs(best_bracket[1] - best_bracket[0]):
            best_bracket = bl

    # Also try starting from x0 with different initial h values for better brackets
    for h_init in [0.1, 0.5, 5.0, 10.0]:
        if not time_ok():
            break
        # Right expansion with different initial h
        a = x
        fa = fx
        b = a + h_init
        fb = f_call(b)
        if math.isfinite(fb) and fa * fb < 0:
            if best_bracket is None or abs(b - a) < abs(best_bracket[1] - best_bracket[0]):
                best_bracket = (a, b, fa, fb)
            break
        # Left expansion with different initial h
        b = a - h_init
        fb = f_call(b)
        if math.isfinite(fb) and fa * fb < 0:
            if best_bracket is None or abs(b - a) < abs(best_bracket[1] - best_bracket[0]):
                best_bracket = (b, a, fb, fa)
            break

    if best_bracket is not None:
        a, b, fa, fb = best_bracket
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Step 2: Hybrid solver - Inverse Quadratic Interpolation with bisection fallback
        # This is essentially Brent's method
        x0_ = a
        x1_ = 0.5 * (a + b)
        x2_ = b
        f0 = fa
        f1 = f_call(x1_)
        f2 = fb

        if not math.isfinite(f1):
            f1 = f_call(0.25 * (a + b))
        if not math.isfinite(f1):
            f1 = f_call(0.75 * (a + b))

        if abs(f1) <= 1e-8:
            return x1_
        if abs(f2) <= 1e-8:
            return x2_
        if abs(f0) <= 1e-8:
            return x0_

        # Track which end has the smaller |f|
        if abs(f1) < abs(f2):
            x2_ = x1_
            f2 = f1
            x1_ = 0.5 * (a + b)
            f1 = f_call(x1_)
            if not math.isfinite(f1):
                f1 = f_call(0.25 * (a + b))

        for _ in range(200):
            if not time_ok():
                break

            if abs(f1) <= 1e-8:
                return x1_
            if abs(f2) <= 1e-8:
                return x2_
            if abs(f0) <= 1e-8:
                return x0_

            # Check convergence
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                # Return the point with smallest |f|
                candidates = [(x0_, abs(f0)), (x1_, abs(f1)), (x2_, abs(f2))]
                candidates.sort(key=lambda t: t[1])
                return candidates[0][0]

            # Try inverse quadratic interpolation
            x_new = None
            if f0 != f1 and f0 != f2 and f1 != f2:
                # IQI formula
                # x_new = x2 - (x2-x1)^2*(x2-x0)*f2*f1 / ((x1-x0)*(x1-x2)*f2*f0 + ...)
                # Standard IQI:
                x_new = (x0 * f1 * f2 / ((f0 - f1) * (f0 - f2)) +
                         x1 * f0 * f2 / ((f1 - f0) * (f1 - f2)) +
                         x2 * f0 * f1 / ((f2 - f0) * (f2 - f1)))
            else:
                # Fall back to secant
                if f2 != f1:
                    x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                else:
                    x_new = 0.5 * (a + b)

            # Check if IQI is acceptable (not too close to endpoints)
            mid1 = 0.5 * (a + x1_)
            mid2 = 0.5 * (x1_ + b)
            if not (mid1 < x_new < mid2) or not math.isfinite(x_new):
                # Fall back to bisection
                x_new = 0.5 * (a + b)

            f_new = f_call(x_new)

            if not math.isfinite(f_new):
                x_new = 0.5 * (a + b)
                f_new = f_call(x_new)
                if not math.isfinite(f_new):
                    break

            if abs(f_new) <= 1e-8:
                return x_new

            # Update bracket and points
            if fa * f_new < 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new

            # Shift points: keep the two points with smallest |f|
            # Order: x0 (older), x1 (middle), x2 (newest, should have smallest |f|)
            if abs(f_new) < abs(f1):
                # x_new is better than x1
                x0_ = x1_
                f0 = f1
                x1_ = x2_
                f1 = f2
                x2_ = x_new
                f2 = f_new
            elif abs(f_new) < abs(f2):
                # x_new is between x1 and x2 in quality
                x0_ = x1_
                f0 = f1
                x1_ = x_new
                f1 = f_new
                # x2 stays
            else:
                # x_new is worse than both
                x0_ = x1_
                f0 = f1
                x1_ = x_new
                f1 = f_new
                # x2 stays

        # Return best point
        candidates = [(x0_, abs(f0)), (x1_, abs(f1)), (x2_, abs(f2))]
        candidates.sort(key=lambda t: t[1])
        return candidates[0][0]

    else:
        # No bracket found. Try Newton's method with finite differences.
        x = float(x0)
        fx = f_call(x)
        
        for _ in range(80):
            if not time_ok():
                break
            if abs(fx) <= 1e-8:
                return x
                
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = f_call(x + h)
            if not math.isfinite(fx_plus):
                h = -h
                fx_plus = f_call(x + h)
            if not math.isfinite(fx_plus):
                break
            d = (fx_plus - fx) / h
            
            if d == 0 or not math.isfinite(d):
                break
                
            x_new = x - fx / d
            
            if not math.isfinite(x_new):
                break
                
            # Damping
            if abs(x_new - x) > 10.0:
                x_new = x + 10.0 * math.copysign(1.0, x_new - x)
                
            x = x_new
            fx = f_call(x)
            
        return x
