"""Root finder for 3 fixed smooth polynomial equations using IQI."""
import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0

    x = float(x0)

    # Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # Try nearby points
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    break
            else:
                continue
            break
        else:
            return x0

    if abs(fx) <= 1e-8:
        return x

    # Get second point with adaptive step
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    try:
        fx2 = f(x2)
    except Exception:
        x2 = x - h
        try:
            fx2 = f(x2)
        except Exception:
            return x

    if not math.isfinite(fx2):
        # Try opposite direction
        x2 = x - h
        try:
            fx2 = f(x2)
        except Exception:
            return x

    if abs(fx2) <= 1e-8:
        return x2

    # History: list of (x, fx)
    history = [(x, fx), (x2, fx2)]
    best_x = x if abs(fx) < abs(fx2) else x2
    best_fx = fx if abs(fx) < abs(fx2) else fx2

    # Main IQI loop
    for _ in range(60):
        if time.time() > deadline:
            break

        # Update best
        for (xi, fxi) in history:
            if abs(fxi) < abs(best_fx):
                best_x = xi
                best_fx = fxi

        if abs(best_fx) <= 1e-8:
            return best_x

        x_new = None

        if len(history) >= 3:
            # Inverse Quadratic Interpolation using last 3 points
            (x0_h, f0_h), (x1_h, f1_h), (x2_h, f2_h) = history[-3], history[-2], history[-1]

            # Check for near-degenerate cases
            df01 = f0_h - f1_h
            df02 = f0_h - f2_h
            df12 = f1_h - f2_h

            if abs(df01) > 1e-30 and abs(df02) > 1e-30 and abs(df12) > 1e-30:
                # Lagrange interpolation for x as function of f, evaluate at f=0
                L0 = (0 - f1_h) * (0 - f2_h) / (df01 * df02)
                L1 = (0 - f0_h) * (0 - f2_h) / ((f1_h - f0_h) * (f1_h - f2_h))
                L2 = (0 - f0_h) * (0 - f1_h) / ((f2_h - f0_h) * (f2_h - f1_h))

                x_new = L0 * x0_h + L1 * x1_h + L2 * x2_h

        if x_new is None and len(history) >= 2:
            # Secant fallback
            (x1_h, f1_h), (x2_h, f2_h) = history[-2], history[-1]
            if abs(f1_h - f2_h) > 1e-30:
                x_new = x2_h - f2_h * (x2_h - x1_h) / (f2_h - f1_h)

        if x_new is None:
            # Last resort: small step from best point
            x_new = best_x + 1e-4 * max(1.0, abs(best_x))

        if not math.isfinite(x_new):
            x_new = best_x + 1e-4

        # Safeguard: limit step size
        max_step = 5.0 * max(1.0, abs(history[-1][0]))
        if abs(x_new - history[-1][0]) > max_step:
            x_new = history[-1][0] + math.copysign(max_step, x_new - history[-1][0])

        # Evaluate
        try:
            fx_new = f(x_new)
        except Exception:
            # Shrink step
            x_new = history[-1][0] + 0.1 * (x_new - history[-1][0])
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(fx_new):
                break

        if not math.isfinite(fx_new):
            x_new = history[-1][0] + 0.1 * (x_new - history[-1][0])
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(fx_new):
                break

        if abs(fx_new) <= 1e-8:
            return x_new

        # Update history: keep last 3
        history.append((x_new, fx_new))
        if len(history) > 3:
            history.pop(0)

        # If we're not making progress, try a different step
        if len(history) == 3:
            last_fx = history[-1][1]
            prev_fx = history[-2][1]
            if abs(last_fx) > abs(prev_fx) * 1.5:
                # Function value increased; try secant between best and current
                (xb, fb) = (best_x, best_fx)
                (xc, fc) = (x_new, fx_new)
                if abs(fb - fc) > 1e-30:
                    x_sec = xc - fc * (xc - xb) / (fc - fb)
                    if math.isfinite(x_sec):
                        try:
                            fx_sec = f(x_sec)
                            if math.isfinite(fx_sec):
                                history = [(xb, fb), (xc, fx_new), (x_sec, fx_sec)]
                                if abs(fx_sec) <= 1e-8:
                                    return x_sec
                        except Exception:
                            pass

    return best_x
