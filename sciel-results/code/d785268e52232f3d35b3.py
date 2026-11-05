"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 148.0  # safety margin before 150s kill

    # Global evaluation cache to avoid redundant calls
    cache = {}
    eval_count = 0

    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            return None
        # Quantize key to avoid float precision issues in dict
        key = float(x)
        if key in cache:
            return cache[key]
        try:
            val = f(key)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            cache[key] = None
            eval_count += 1
            return None
        if not math.isfinite(val):
            cache[key] = None
            eval_count += 1
            return None
        cache[key] = val
        eval_count += 1
        return val

    def check_time():
        return time.monotonic() < DEADLINE and eval_count < MAX_EVALS - 2

    def is_good(fx):
        return fx is not None and abs(fx) <= TOL

    # Track best result globally
    best_x = float(x0)
    best_f = call_f(best_x)
    best_abs = abs(best_f) if best_f is not None else float('inf')

    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if fx is not None:
            af = abs(fx)
            if af < best_abs:
                best_abs = af
                best_x = x
                best_f = fx

    # Illinois algorithm (robust secant variant for ill-conditioned problems)
    def illinois(x0, x1, max_iter=80):
        if not check_time():
            return None, None
        a = float(x0)
        b = float(x1)
        fa = call_f(a)
        fb = call_f(b)
        update_best(a, fa)
        update_best(b, fb)
        if is_good(fa):
            return a, fa
        if is_good(fb):
            return b, fb
        if fa is None or fb is None:
            return None, None

        # Ensure fa and fb have opposite signs or just proceed
        for _ in range(max_iter):
            if not check_time():
                break
            if fa * fb > 0:
                # No sign change; try to find one by expanding
                break

            # Illinois: if f(a) and f(b) have same sign after update, halve the interval on the side that just moved
            # Standard Illinois:
            # x_new = (a*fb - b*fa) / (fb - fa)
            denom = fb - fa
            if abs(denom) < 1e-30:
                break
            x_new = (a * fb - b * fa) / denom
            if not math.isfinite(x_new):
                break

            # Damping
            max_step = 10 * max(1.0, abs(a), abs(b))
            if abs(x_new - a) > max_step:
                x_new = a + math.copysign(max_step, x_new - a)
            if abs(x_new - b) > max_step:
                x_new = b + math.copysign(max_step, x_new - b)

            fx_new = call_f(x_new)
            update_best(x_new, fx_new)
            if fx_new is None:
                break
            if is_good(fx_new):
                return x_new, fx_new

            # Illinois update: replace the point with the larger |f|
            if abs(fa) > abs(fb):
                # Replace a with midpoint of a and x_new
                a = (a + x_new) / 2.0
                fa = call_f(a)
                update_best(a, fa)
                if fa is None:
                    break
                if is_good(fa):
                    return a, fa
            else:
                # Replace b with midpoint of b and x_new
                b = (b + x_new) / 2.0
                fb = call_f(b)
                update_best(b, fb)
                if fb is None:
                    break
                if is_good(fb):
                    return b, fb

        return None, None

    # Safeguarded Newton (with bisection fallback)
    def safeguarded_newton(x0, max_iter=50):
        if not check_time():
            return None, None
        x = float(x0)
        fx = call_f(x)
        update_best(x, fx)
        if is_good(fx):
            return x, fx
        if fx is None:
            return None, None

        # Estimate bracket
        h = 1e-4 * max(1.0, abs(x))
        a, b = x - h, x + h
        fa = call_f(a)
        fb = call_f(b)
        update_best(a, fa)
        update_best(b, fb)
        if is_good(fa):
            return a, fa
        if is_good(fb):
            return b, fb

        for _ in range(max_iter):
            if not check_time():
                break

            # Derivative estimate
            h_d = 1e-6 * max(1.0, abs(x))
            fx_plus = call_f(x + h_d)
            fx_minus = call_f(x - h_d)
            if fx_plus is None or fx_minus is None:
                break
            d = (fx_plus - fx_minus) / (2 * h_d)
            if not math.isfinite(d) or abs(d) < 1e-15:
                break

            x_new = x - fx / d
            if not math.isfinite(x_new):
                break

            # Damping
            max_step = 5 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + math.copysign(max_step, x_new - x)

            fx_new = call_f(x_new)
            update_best(x_new, fx_new)
            if fx_new is None:
                break
            if is_good(fx_new):
                return x_new, fx_new

            # Check if we're making progress
            if abs(fx_new) < abs(fx) * 0.9:
                x = x_new
                fx = fx_new
            else:
                # Try bisection step toward the better side
                if fa is not None and fb is not None and fa * fb < 0:
                    # We have a bracket; use bisection
                    mid = (a + b) / 2.0
                    fmid = call_f(mid)
                    update_best(mid, fmid)
                    if fmid is None:
                        break
                    if is_good(fmid):
                        return mid, fmid
                    if fa * fmid < 0:
                        b = mid
                        fb = fmid
                    else:
                        a = mid
                        fa = fmid
                    x = (a + b) / 2.0
                    fx = call_f(x)
                    update_best(x, fx)
                    if is_good(fx):
                        return x, fx
                else:
                    # No bracket; take a smaller Newton step
                    x_trial = x - 0.5 * (fx / d)
                    fx_trial = call_f(x_trial)
                    update_best(x_trial, fx_trial)
                    if fx_trial is None:
                        break
                    if is_good(fx_trial):
                        return x_trial, fx_trial
                    if abs(fx_trial) < abs(fx):
                        x = x_trial
                        fx = fx_trial
                    else:
                        break

        return None, None

    # Simple secant (no Illinois halving, pure secant)
    def pure_secant(x0, x1, max_iter=80):
        if not check_time():
            return None, None
        a = float(x0)
        b = float(x1)
        fa = call_f(a)
        fb = call_f(b)
        update_best(a, fa)
        update_best(b, fb)
        if is_good(fa):
            return a, fa
        if is_good(fb):
            return b, fb
        if fa is None or fb is None:
            return None, None

        for _ in range(max_iter):
            if not check_time():
                break
            denom = fb - fa
            if abs(denom) < 1e-30:
                break
            x_new = b - fb * (b - a) / denom
            if not math.isfinite(x_new):
                break

            max_step = 10 * max(1.0, abs(a), abs(b))
            if abs(x_new - b) > max_step:
                x_new = b + math.copysign(max_step, x_new - b)

            fx_new = call_f(x_new)
            update_best(x_new, fx_new)
            if fx_new is None:
                break
            if is_good(fx_new):
                return x_new, fx_new

            # Accept if improving
            if abs(fx_new) < abs(fb):
                a = b
                fa = fb
                b = x_new
                fb = fx_new
            else:
                # Try midpoint
                mid = (a + b) / 2.0
                fmid = call_f(mid)
                update_best(mid, fmid)
                if fmid is None:
                    break
                if is_good(fmid):
                    return mid, fmid
                if abs(fmid) < abs(fb):
                    a = b
                    fa = fb
                    b = mid
                    fb = fmid
                else:
                    b = mid
                    fb = fmid

        return None, None

    # Generate diverse starting points
    base = float(x0)
    scale = max(1.0, abs(base))

    # Multi-start points: diverse scales and directions
    start_points = []
    # Small perturbations
    for s in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        start_points.append(base + s * scale)
        start_points.append(base - s * scale)
    # Also include the original
    start_points.append(base)

    # Remove duplicates (approximately)
    unique_starts = []
    for sp in start_points:
        if not math.isfinite(sp):
            continue
        is_dup = False
        for us in unique_starts:
            if abs(sp - us) < 1e-12 * max(1.0, abs(sp), abs(us)):
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(sp)

    # Launch multiple independent tracks
    # Each track is a (method, args) tuple
    # We'll run them in a round-robin fashion to balance evaluation budget

    # Track 1: Illinois from x0 and x0+1e-4*scale
    # Track 2: Safeguarded Newton from x0
    # Track 3: Pure secant from x0 and x0+1e-3*scale
    # Track 4: Illinois from x0+1e-2*scale and x0+1e-2*scale+1e-3*scale
    # Track 5: Safeguarded Newton from x0+1e-3*scale
    # etc.

    # Define tracks as (method_name, x0, x1) for two-point methods, or (method_name, x0) for one-point
    tracks = []
    # Illinois tracks (need two points)
    h1 = 1e-4 * scale
    tracks.append(('illinois', base, base + h1))
    h2 = 1e-3 * scale
    tracks.append(('illinois', base + h2, base + h2 + 1e-4 * scale))
    h3 = 1e-2 * scale
    tracks.append(('illinois', base + h3, base + h3 + 1e-3 * scale))
    h4 = 0.1 * scale
    tracks.append(('illinois', base - h4, base - h4 + 1e-3 * scale))

    # Safeguarded Newton tracks (one point)
    tracks.append(('newton', base, None))
    tracks.append(('newton', base + 1e-3 * scale, None))
    tracks.append(('newton', base - 1e-3 * scale, None))
    tracks.append(('newton', base + 1e-2 * scale, None))

    # Pure secant tracks
    tracks.append(('secant', base, base + 1e-3 * scale))
    tracks.append(('secant', base + 1e-2 * scale, base + 1e-2 * scale + 1e-4 * scale))
    tracks.append(('secant', base - 1e-2 * scale, base - 1e-2 * scale + 1e-4 * scale))

    # We'll run these in a round-robin, giving each track a few iterations before moving on
    # To keep it simple, we'll just run each track to completion or until budget is exhausted,
    # but interleave them to ensure diversity

    # Actually, a simpler approach: run each track sequentially but with a time/eval budget per track
    # If a track fails quickly, move to the next

    # Let's do a hybrid: run the most promising tracks first
    # Priority: Illinois (robust), then Newton (fast), then secant

    # Reorder tracks
    tracks.sort(key=lambda t: (
        0 if t[0] == 'illinois' else 1 if t[0] == 'newton' else 2
    ))

    for method, xa, xb in tracks:
        if not check_time():
            break
        if method == 'illinois':
            x_res, f_res = illinois(xa, xb)
            if x_res is not None and is_good(f_res):
                return x_res
        elif method == 'newton':
            x_res, f_res = safeguarded_newton(xa)
            if x_res is not None and is_good(f_res):
                return x_res
        elif method == 'secant':
            x_res, f_res = pure_secant(xa, xb)
            if x_res is not None and is_good(f_res):
                return x_res

    # If all tracks fail, try a final broad sweep
    if check_time():
        # Try Illinois with wider brackets
        for h in [1.0, 10.0, 100.0, 1000.0]:
            if not check_time():
                break
            x_res, f_res = illinois(base - h, base + h)
            if x_res is not None and is_good(f_res):
                return x_res

    # Return best found
    return best_x

# EVOLVE-BLOCK-END
