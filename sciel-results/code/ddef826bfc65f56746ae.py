"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: kill after ~110s to stay well under 120s wall-clock.
    deadline = time.time() + 110.0

    x = float(x0)
    # Best solution seen so far (lowest |f(x)|)
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    call_count = 1

    if best_abs <= 1e-8:
        return best_x

    # We'll maintain a list of (x, fx) pairs for interpolation.
    # Start with just x0.
    points = [(x, best_fx)]

    # We'll use a hybrid strategy:
    # 1. Try to recover polynomial coefficients by evaluating at N+1 points (N=5 assumed).
    #    If that works (residual small), use numpy.roots to find all roots and pick closest to x0.
    # 2. If polynomial recovery fails or gives bad results, fall back to robust iterative
    #    methods: inverse cubic / secant / bisection.
    #
    # But first, let's try a quick local approach with few calls, then escalate.

    # --- Phase 1: Try polynomial coefficient recovery (degree up to 5) ---
    # Evaluate at x0, x0+1, x0+2, x0+3, x0+4, x0+5 (6 points for degree 5)
    # Actually, we already have x0. Evaluate 5 more points.
    # But we need to be careful: if the polynomial has high degree or is not degree <= 5,
    # this won't work. We'll check the residual.

    # Let's try degree 2 first (3 points), then degree 3, 4, 5.
    # Start with 3 points: x0, x0+1, x0-1 (symmetric for better conditioning)
    # Actually, let's use x0, x0+1, x0+2 for simplicity, then check.

    # Better approach: try multiple degrees and see which gives smallest residual.
    # For degree d, we need d+1 points.

    # Let's evaluate at x0, x0+0.5, x0+1.0, x0+1.5, x0+2.0, x0+2.5 (6 points, supports up to degree 5)
    # But we already have x0. So 5 more calls.

    # Actually, let me be more strategic. Let's try:
    # First, do a few secant/Newton steps to get close to a root, then polish.
    # This is more robust than assuming polynomial degree.

    # --- Revised strategy: Robust iterative method with interpolation ---
    # Use a combination of:
    # - Inverse quadratic/cubic interpolation when available
    # - Secant method
    # - Bisection when bracket exists
    # - Newton with finite difference derivative as fallback

    # Let's implement a robust root finder that minimizes function calls.

    # State:
    # - points: list of (x, fx)
    # - We'll try to find a bracket [a, b] where f(a)*f(b) < 0
    # - Then use bisection + interpolation

    # First, try to find a bracket by expanding from x0.
    a = x
    fa = best_fx
    b = x + 1.0
    fb = f(b)
    call_count += 1
    points.append((b, fb))

    if abs(fb) < best_abs:
        best_x = b
        best_abs = abs(fb)
        best_fx = fb
    if best_abs <= 1e-8:
        return best_x

    # Try to find a bracket
    # Expand b in both directions
    step = 1.0
    bracket_found = False
    a_bracket = None
    b_bracket = None

    # Check if x0 and x0+1 bracket a root
    if fa * fb < 0:
        bracket_found = True
        a_bracket = a
        b_bracket = b
    else:
        # Try expanding
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Try x0 - step
            c = x - step
            fc = f(c)
            call_count += 1
            points.append((c, fc))
            if abs(fc) < best_abs:
                best_x = c
                best_abs = abs(fc)
                best_fx = fc
            if best_abs <= 1e-8:
                return best_x

            if fa * fc < 0:
                bracket_found = True
                a_bracket = c
                b_bracket = x
                break
            if fc * fb < 0:
                bracket_found = True
                a_bracket = x
                b_bracket = b
                break

            # Try x0 + step*2 (expand right)
            d = x + step * 2
            fd = f(d)
            call_count += 1
            points.append((d, fd))
            if abs(fd) < best_abs:
                best_x = d
                best_abs = abs(fd)
                best_fx = fd
            if best_abs <= 1e-8:
                return best_x

            if fa * fd < 0:
                bracket_found = True
                a_bracket = x
                b_bracket = d
                break
            if fb * fd < 0:
                bracket_found = True
                a_bracket = b
                b_bracket = d
                break

            step *= 2
            if step > 1e10:
                break

    # If we have a bracket, use bisection + interpolation
    if bracket_found:
        lo = a_bracket
        hi = b_bracket
        f_lo = f(lo) if (lo, f(lo)) not in points else [fx for px, fx in points if px == lo][0]
        # Actually, let's just re-evaluate if needed, but we should have it.
        # Let me restructure to keep track of function values properly.

        # Let's use a cleaner approach:
        # Maintain a dict of x -> fx for evaluated points
        pass

    # Let me rewrite this more cleanly.
    
    # --- Clean implementation ---
    x = float(x0)
    best_x = x
    best_abs = float('inf')
    evals = {}  # x -> fx

    def eval_f(xv):
        nonlocal call_count, best_x, best_abs
        if xv in evals:
            return evals[xv]
        fx = f(xv)
        evals[xv] = fx
        call_count += 1
        afx = abs(fx)
        if afx < best_abs:
            best_abs = afx
            best_x = xv
        return fx

    eval_f(x)

    if best_abs <= 1e-8:
        return best_x

    # Try to find a bracket
    step = 1.0
    lo = x
    hi = x + step
    f_lo = eval_f(lo)
    f_hi = eval_f(hi)

    if abs(f_hi) < best_abs:
        best_abs = abs(f_hi)
        best_x = hi

    if best_abs <= 1e-8:
        return best_x

    bracket = None
    if f_lo * f_hi < 0:
        bracket = (lo, hi, f_lo, f_hi)
    else:
        for _ in range(60):
            if time.time() > deadline - 5:
                break
            # Expand left
            c = x - step
            f_c = eval_f(c)
            if f_lo * f_c < 0:
                bracket = (c, x, f_c, f_lo)
                break
            if f_c * f_hi < 0:
                bracket = (x, hi, f_lo, f_hi)
                break
            # Expand right
            d = x + step * 2
            f_d = eval_f(d)
            if f_lo * f_d < 0:
                bracket = (x, d, f_lo, f_d)
                break
            if f_hi * f_d < 0:
                bracket = (hi, d, f_hi, f_d)
                break
            step *= 2
            if step > 1e12:
                break

    if bracket is not None:
        a, b, fa, fb = bracket
        # Use bisection with interpolation
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            if best_abs <= 1e-10:
                break
            # Try inverse quadratic interpolation if we have 3 points
            # For now, use secant + bisection hybrid
            mid = (a + b) / 2.0
            f_mid = eval_f(mid)
            if abs(f_mid) < best_abs:
                best_abs = abs(f_mid)
                best_x = mid
            if best_abs <= 1e-10:
                break
            if fa * f_mid <= 0:
                b = mid
                fb = f_mid
            else:
                a = mid
                fa = f_mid
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        return best_x

    # No bracket found; use secant method
    # We have at least 2 points: x and x+1 (or more)
    # Sort points by x
    sorted_pts = sorted(evals.items(), key=lambda kv: kv[0])
    if len(sorted_pts) < 2:
        return best_x

    # Use the two closest points to x0 for secant
    # Actually, use the two most recent or closest to x
    # Let's just use x and x+1
    x1 = x
    x2 = x + 1.0
    f1 = evals.get(x1)
    f2 = evals.get(x2)
    if f1 is None or f2 is None:
        return best_x

    for _ in range(50):
        if time.time() > deadline - 2:
            break
        if best_abs <= 1e-10:
            break
        if f2 - f1 == 0:
            break
        x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        f_new = eval_f(x_new)
        if best_abs <= 1e-10:
            break
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        if abs(x_new - x2) < 1e-14 * max(1.0, abs(x_new)):
            break

    return best_x
# EVOLVE-BLOCK-END
