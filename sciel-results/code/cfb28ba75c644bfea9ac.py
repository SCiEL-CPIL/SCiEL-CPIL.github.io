"""Root finder for smooth polynomial equations - secant-based hybrid."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.monotonic()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    x = float(x0)

    # Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # Try nearby points to find a finite value
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                xt = x + sign * dx
                try:
                    fxt = f(xt)
                except Exception:
                    continue
                if math.isfinite(fxt):
                    x, fx = xt, fxt
                    break
            if math.isfinite(fx):
                break
        if not math.isfinite(fx):
            return x0

    if abs(fx) <= 1e-8:
        return x

    # Track best
    best_x, best_f = x, fx

    # Phase 1: Try to find a bracket via expansion (limit to 15 iterations to save calls)
    a, b = x, x
    fa, fb = fx, fx
    step = 1.0
    bracketed = False

    for i in range(15):
        if time.monotonic() > deadline:
            break
        a = x - step
        b = x + step
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            step *= 2.0
            continue

        if not (math.isfinite(fa) and math.isfinite(fb)):
            step *= 2.0
            continue

        if abs(fa) < abs(best_f):
            best_x, best_f = a, fa
        if abs(fb) < abs(best_f):
            best_x, best_f = b, fb

        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb < 0.0:
            bracketed = True
            break
        step *= 2.0

    if bracketed:
        # Brent-like method: combine bisection, secant, and inverse quadratic interpolation
        lo, hi = a, b
        flo, fhi = fa, fb
        # Ensure |flo| <= |fhi| so we keep the one with smaller magnitude as 'lo'
        if abs(flo) > abs(fhi):
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        c = lo
        fc = flo
        e = lo - hi  # last interpolation step

        for it in range(80):
            if time.monotonic() > deadline:
                break

            if abs(flo) <= 1e-8:
                return lo
            if abs(fhi) <= 1e-8:
                return hi

            # Convergence check
            if abs(hi - lo) < 1e-14:
                break

            # Inverse quadratic interpolation
            if (c != lo and c != hi and fc != 0.0):
                t = flo / fhi
                if t != 1.0:
                    q = 1.0 - t
                    r = (hi - c) * (flo / fc) if fc != 0 else 0.0
                    p = r * q - (hi - lo) * t
                    e_old = e
                    e = r * q - (hi - lo) * t
                    if p > 0:
                        e = -e
                    p = p / (2.0 * abs(q))
                else:
                    p = e
                    e = e

                # Check if IQUO step is acceptable
                ok = True
                # p must be greater than half the smallest allowable step
                if not (p > 0.5 * min(2.0 * (hi - lo) * (abs(flo) - abs(fhi)) / abs(fhi), abs(hi - lo))):
                    ok = False
                # p must be less than the last step
                if not (p < abs(e)):
                    ok = False
                # p must be within the bracket
                if not ((hi - lo) * p < 0.5 * (hi - lo) * (abs(flo) - abs(fhi)) / abs(fhi)):
                    ok = False

                if ok:
                    d = e
                    e = p
                else:
                    # Bisection
                    e = 0.5 * (hi - lo)
                    d = e
            else:
                # Secant step
                if flo != fhi:
                    e = 0.5 * (hi - lo)
                    d = (hi - lo) * (flo / (flo - fhi))
                else:
                    e = 0.5 * (hi - lo)
                    d = e

            # If d is too small, use bisection
            if abs(d) < 1e-14:
                x_new = lo + d
            else:
                x_new = lo + d

            # Ensure x_new is within bracket
            if x_new <= lo or x_new >= hi:
                x_new = 0.5 * (lo + hi)

            try:
                f_new = f(x_new)
            except Exception:
                x_new = 0.5 * (lo + hi)
                try:
                    f_new = f(x_new)
                except Exception:
                    break

            if not math.isfinite(f_new):
                x_new = 0.5 * (lo + hi)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break

            if abs(f_new) <= 1e-8:
                return x_new

            # Update best
            if abs(f_new) < abs(best_f):
                best_x, best_f = x_new, f_new

            # Update bracket
            if f_new * flo < 0:
                hi, fhi = x_new, f_new
            else:
                c, fc = lo, flo
                lo, flo = x_new, f_new

            # Keep |flo| <= |fhi|
            if abs(flo) > abs(fhi):
                lo, hi = hi, lo
                flo, fhi = fhi, flo

        # Return best from bracket
        if abs(flo) <= abs(fhi):
            return lo
        else:
            return hi
    else:
        # No bracket found: use safeguarded secant/Newton hybrid
        # This approach works well for smooth polynomials even without a sign change
        
        # Try secant method from x0 and a nearby point
        x1 = x + 1.0
        try:
            f1 = f(x1)
        except Exception:
            f1 = None
        
        if f1 is not None and math.isfinite(f1):
            if abs(f1) < abs(best_f):
                best_x, best_f = x1, f1
            if abs(f1) <= 1e-8:
                return x1
            
            # Secant iterations
            x_curr, f_curr = x1, f1
            x_prev, f_prev = x, fx
            
            for it in range(60):
                if time.monotonic() > deadline:
                    break
                if abs(f_curr) <= 1e-8:
                    return x_curr
                
                if f_prev == f_curr:
                    break
                    
                x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                
                # Safeguard: don't go too far
                max_step = max(1.0, abs(x_curr)) * 5.0
                if abs(x_new - x_curr) > max_step:
                    x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)
                
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break
                
                if abs(f_new) < abs(f_curr):
                    x_prev, f_prev = x_curr, f_curr
                    x_curr, f_curr = x_new, f_new
                    if abs(f_curr) < abs(best_f):
                        best_x, best_f = x_curr, f_curr
                else:
                    # Try bisection-like step toward better point
                    if abs(f_prev) < abs(f_curr):
                        x_curr, f_curr = x_prev, f_prev
                        x_prev, f_prev = x_new, f_new
                    else:
                        # Damped step
                        x_new = 0.5 * (x_curr + x_prev)
                        try:
                            f_new = f(x_new)
                        except Exception:
                            break
                        if math.isfinite(f_new):
                            if abs(f_new) < abs(f_curr):
                                x_prev, f_prev = x_curr, f_curr
                                x_curr, f_curr = x_new, f_new
                                if abs(f_curr) < abs(best_f):
                                    best_x, best_f = x_curr, f_curr
                            else:
                                break
                        else:
                            break
        
        # Final polish: try a few more secant/Newton steps from best point
        if abs(best_f) > 1e-8 and abs(best_f) < 1.0:
            x_curr = best_x
            f_curr = best_f
            for it in range(20):
                if time.monotonic() > deadline:
                    break
                if abs(f_curr) <= 1e-8:
                    return x_curr
                
                # Finite difference derivative
                h = 1e-7 * max(1.0, abs(x_curr))
                try:
                    f_xh = f(x_curr + h)
                except Exception:
                    break
                if not math.isfinite(f_xh):
                    break
                
                d = (f_xh - f_curr) / h
                if d == 0 or not math.isfinite(d):
                    break
                
                x_new = x_curr - f_curr / d
                
                # Safeguard
                max_step = max(1.0, abs(x_curr)) * 2.0
                if abs(x_new - x_curr) > max_step:
                    x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)
                
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break
                
                if abs(f_new) < abs(f_curr):
                    x_curr, f_curr = x_new, f_new
                    if abs(f_curr) < abs(best_f):
                        best_x, best_f = x_curr, f_curr
                else:
                    # Damped step
                    x_new = 0.5 * (x_curr + x_new)
                    try:
                        f_new = f(x_new)
                    except Exception:
                        break
                    if math.isfinite(f_new) and abs(f_new) < abs(f_curr):
                        x_curr, f_curr = x_new, f_new
                        if abs(f_curr) < abs(best_f):
                            best_x, best_f = x_curr, f_curr
                    else:
                        break

    return best_x
# EVOLVE-BLOCK-END
