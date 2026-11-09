import math
import time

def find_root(f, x0):
    """
    Finds a root of f near x0.
    Uses local polynomial interpolation (cubic/quadratic Lagrange) to predict
    the nearest zero-crossing, then safeguards with a hybrid Newton-Bisection
    refinement.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    tol_final = 1e-8

    x0 = float(x0)

    # Evaluate at x0
    try:
        f0 = f(x0)
    except Exception:
        f0 = float('nan')

    if math.isfinite(f0) and abs(f0) <= tol_final:
        return x0

    # Handle non-finite f0
    if not math.isfinite(f0):
        base_step = 1e-4 * max(1.0, abs(x0))
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64]:
            if time.time() > deadline:
                break
            x_try = x0 + s * base_step
            try:
                f_try = f(x_try)
            except Exception:
                continue
            if math.isfinite(f_try) and abs(f_try) <= tol_final:
                return x_try
            if math.isfinite(f_try):
                x0, f0 = x_try, f_try
                break
        if not math.isfinite(f0):
            return x0

    # Collect initial points for interpolation
    base_step = 1e-6 * max(1.0, abs(x0))
    xs = [x0]
    fs = [f0]
    for s in [1, -1, 2, -2, 4, -4]:
        if time.time() > deadline:
            break
        x_try = x0 + s * base_step
        try:
            f_try = f(x_try)
        except Exception:
            continue
        if math.isfinite(f_try):
            xs.append(x_try)
            fs.append(f_try)
            if abs(f_try) <= tol_final:
                return x_try

    best_x = xs[0]
    best_f = abs(fs[0])
    for i in range(1, len(xs)):
        if abs(fs[i]) < best_f:
            best_x = xs[i]
            best_f = abs(fs[i])

    # Try to find a bracket among the initial points
    a, b, fa, fb = None, None, None, None
    min_width = float('inf')
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            f1, f2 = fs[i], fs[j]
            if f1 * f2 < 0:
                width = abs(xs[j] - xs[i])
                if width < min_width:
                    min_width = width
                    a, b = xs[i], xs[j]
                    fa, fb = f1, f2
            elif f1 == 0:
                return xs[i]
            elif f2 == 0:
                return xs[j]

    # If no bracket found, try expanding
    if a is None:
        current_range = 20.0
        found = False
        while current_range < 1000.0 and not found:
            if time.time() > deadline:
                break
            x_left = x0 - current_range
            x_right = x0 + current_range
            try:
                fl = f(x_left)
                fr = f(x_right)
            except Exception:
                current_range *= 2
                continue
            if math.isfinite(fl) and math.isfinite(fr):
                if fl * fr < 0:
                    a, b = x_left, x_right
                    fa, fb = fl, fr
                    found = True
            current_range *= 2

    if a is None:
        return best_x

    # Use interpolation to predict the root
    # Select up to 4 points closest to the bracket center
    center = (a + b) / 2.0
    # Sort points by distance to center
    points_sorted = sorted(zip(xs, fs), key=lambda p: abs(p[0] - center))
    interp_points = points_sorted[:4]

    # Fit a polynomial and find its root
    x_root = None
    if len(interp_points) >= 2:
        # Use Lagrange interpolation to find the root
        # For 2 points: linear
        if len(interp_points) == 2:
            x1, f1 = interp_points[0]
            x2, f2 = interp_points[1]
            if f1 != f2:
                x_root = x1 - f1 * (x2 - x1) / (f2 - f1)
        elif len(interp_points) == 3:
            # Quadratic interpolation
            x1, f1 = interp_points[0]
            x2, f2 = interp_points[1]
            x3, f3 = interp_points[2]
            # Solve for root of quadratic through (x1,f1), (x2,f2), (x3,f3)
            # Using Lagrange form
            def lagrange_quad(x):
                L1 = (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))
                L2 = (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))
                L3 = (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2))
                return L1 * f1 + L2 * f2 + L3 * f3
            # Find root of quadratic
            # ax^2 + bx + c = 0
            # Compute coefficients
            a_coef = (f1 / ((x1 - x2) * (x1 - x3)) + 
                      f2 / ((x2 - x1) * (x2 - x3)) + 
                      f3 / ((x3 - x1) * (x3 - x2)))
            b_coef = (f1 * (-(x2 + x3) + 2*x1) / ((x1 - x2) * (x1 - x3)) + 
                      f2 * (-(x1 + x3) + 2*x2) / ((x2 - x1) * (x2 - x3)) + 
                      f3 * (-(x1 + x2) + 2*x3) / ((x3 - x1) * (x3 - x2)))
            c_coef = (f1 * x2 * x3 / ((x1 - x2) * (x1 - x3)) + 
                      f2 * x1 * x3 / ((x2 - x1) * (x2 - x3)) + 
                      f3 * x1 * x2 / ((x3 - x1) * (x3 - x2)))
            if abs(a_coef) > 1e-15:
                disc = b_coef * b_coef - 4 * a_coef * c_coef
                if disc >= 0:
                    sqrt_disc = math.sqrt(disc)
                    r1 = (-b_coef + sqrt_disc) / (2 * a_coef)
                    r2 = (-b_coef - sqrt_disc) / (2 * a_coef)
                    # Choose root closest to center
                    if abs(r1 - center) < abs(r2 - center):
                        x_root = r1
                    else:
                        x_root = r2
            else:
                if abs(b_coef) > 1e-15:
                    x_root = -c_coef / b_coef
        else:
            # Cubic interpolation
            x1, f1 = interp_points[0]
            x2, f2 = interp_points[1]
            x3, f3 = interp_points[2]
            x4, f4 = interp_points[3]
            # Use Newton's method on the cubic to find root
            # Start from center
            x_newton = center
            for _ in range(10):
                # Evaluate cubic and derivative at x_newton
                L1 = ((x_newton - x2) * (x_newton - x3) * (x_newton - x4)) / ((x1 - x2) * (x1 - x3) * (x1 - x4))
                L2 = ((x_newton - x1) * (x_newton - x3) * (x_newton - x4)) / ((x2 - x1) * (x2 - x3) * (x2 - x4))
                L3 = ((x_newton - x1) * (x_newton - x2) * (x_newton - x4)) / ((x3 - x1) * (x3 - x2) * (x3 - x4))
                L4 = ((x_newton - x1) * (x_newton - x2) * (x_newton - x3)) / ((x4 - x1) * (x4 - x2) * (x4 - x3))
                val = L1 * f1 + L2 * f2 + L3 * f3 + L4 * f4
                # Derivative
                dL1 = ((x_newton - x3) * (x_newton - x4) + (x_newton - x2) * (x_newton - x4) + (x_newton - x2) * (x_newton - x3)) / ((x1 - x2) * (x1 - x3) * (x1 - x4))
                dL2 = ((x_newton - x3) * (x_newton - x4) + (x_newton - x1) * (x_newton - x4) + (x_newton - x1) * (x_newton - x3)) / ((x2 - x1) * (x2 - x3) * (x2 - x4))
                dL3 = ((x_newton - x2) * (x_newton - x4) + (x_newton - x1) * (x_newton - x4) + (x_newton - x1) * (x_newton - x2)) / ((x3 - x1) * (x3 - x2) * (x3 - x4))
                dL4 = ((x_newton - x2) * (x_newton - x3) + (x_newton - x1) * (x_newton - x3) + (x_newton - x1) * (x_newton - x2)) / ((x4 - x1) * (x4 - x2) * (x4 - x3))
                dval = dL1 * f1 + dL2 * f2 + dL3 * f3 + dL4 * f4
                if abs(dval) < 1e-15:
                    break
                x_newton_new = x_newton - val / dval
                if not math.isfinite(x_newton_new):
                    break
                if abs(x_newton_new - x_newton) < 1e-12:
                    x_newton = x_newton_new
                    break
                x_newton = x_newton_new
            x_root = x_newton

    # If interpolation gave a root within the bracket, use it as starting point
    if x_root is not None and a < x_root < b:
        x_curr = x_root
        try:
            f_curr = f(x_curr)
        except Exception:
            f_curr = float('nan')
        if math.isfinite(f_curr):
            if abs(f_curr) <= tol_final:
                return x_curr
            # Update bracket if needed
            if fa * f_curr < 0:
                b = x_curr
                fb = f_curr
            elif fb * f_curr < 0:
                a = x_curr
                fa = f_curr
    else:
        # Start from midpoint
        x_curr = (a + b) / 2.0
        try:
            f_curr = f(x_curr)
        except Exception:
            f_curr = float('nan')
        if not math.isfinite(f_curr):
            return best_x
        if abs(f_curr) <= tol_final:
            return x_curr
        if fa * f_curr < 0:
            b = x_curr
            fb = f_curr
        elif fb * f_curr < 0:
            a = x_curr
            fa = f_curr

    # Hybrid Newton-Bisection refinement
    for _ in range(50):
        if time.time() > deadline:
            break
        if abs(f_curr) < 1e-12:
            return x_curr

        # Estimate derivative using Central Difference
        h = 1e-5 * max(1.0, abs(x_curr))
        try:
            f_plus = f(x_curr + h)
            f_minus = f(x_curr - h)
        except Exception:
            f_plus = f(x_curr + h)
            f_minus = f_curr

        df = (f_plus - f_minus) / (2 * h)

        if df == 0 or not math.isfinite(df):
            x_next = (a + b) / 2.0
        else:
            x_next = x_curr - f_curr / df

        if x_next < a or x_next > b:
            x_next = (a + b) / 2.0

        try:
            f_next = f(x_next)
        except Exception:
            x_next = (a + b) / 2.0
            f_next = f(x_next)

        if not math.isfinite(f_next):
            x_next = (a + b) / 2.0
            f_next = f(x_next)

        # Update Bracket
        if f_curr * f_next < 0:
            a, b = x_curr, x_next
            fa, fb = f_curr, f_next
        else:
            if fa * f_next < 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next

        x_curr, f_curr = x_next, f_next

    return x_curr
