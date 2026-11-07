# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 110.0  # safety margin vs 150s wall-clock
    x0 = float(x0)
    fx0 = f(x0)
    if not math.isfinite(fx0):
        return x0
    if abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_f = fx0
    best_abs = abs(fx0)

    # ---- Phase 1: bidirectional bracket search (geometric expansion) ----
    a = x0
    fa = fx0
    b = None
    fb = None

    h = 1e-4
    max_h = 1e6
    # Try to find a sign change on either side
    while h <= max_h and time.time() < deadline:
        # positive direction
        xp = x0 + h
        fxp = f(xp)
        if math.isfinite(fxp):
            if abs(fxp) < best_abs:
                best_abs = abs(fxp)
                best_x = xp
            if fa * fxp < 0:
                a, fa = x0, fa
                b, fb = xp, fxp
                break
        # negative direction
        xn = x0 - h
        fxn = f(xn)
        if math.isfinite(fxn):
            if abs(fxn) < best_abs:
                best_abs = abs(fxn)
                best_x = xn
            if fa * fxn < 0:
                a, fa = xn, fxn
                b, fb = x0, fa
                break
        h *= 2.0

    # If no bracket found, return best point
    if b is None:
        if time.time() > deadline - 1.0:
            return best_x
        # Try a few more scattered points as a last resort
        for scale in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
            if time.time() > deadline - 1.0:
                break
            for sign in (1, -1):
                xc = x0 + sign * scale
                fxc = f(xc)
                if math.isfinite(fxc) and abs(fxc) < best_abs:
                    best_abs = abs(fxc)
                    best_x = xc
                    if abs(fxc) <= 1e-8:
                        return xc
        return best_x

    # ---- Phase 2: refine with IQI (inverse quadratic interpolation) ----
    # We have [a, b] with fa*fb < 0
    # Keep 3 points for IQI: (a, fa), (b, fb), and a mid-point
    # Start with the two bracket points, add a third via bisection
    xa, fa = a, fa
    xb, fb = b, fb

    # Ensure |fa| <= |fb| so we interpolate toward the smaller value
    if abs(fa) > abs(fb):
        xa, xb = xb, xa
        fa, fb = fb, fa

    # Third point: bisection of current bracket
    xm = 0.5 * (xa + xb)
    fm = f(xm)
    if not math.isfinite(fm):
        return best_x
    if abs(fm) < best_abs:
        best_abs = abs(fm)
        best_x = xm
    if abs(fm) <= 1e-8:
        return xm

    # Now we have 3 points: (xa,fa), (xb,fb), (xm,fm)
    # Use IQI to find next point, with bisection safeguard
    for _ in range(60):
        if time.time() > deadline - 1.0:
            break

        # Determine bracket from the 3 points: find the pair with sign change
        # Points: (xa,fa), (xm,fm), (xb,fb)
        # The bracket is between the outer points that have opposite signs
        # Since fa*fb<0 initially, and we maintain this, xa and xb are the bracket
        # But after IQI, the new point may be inside, so we need to update

        # Check if any point satisfies tolerance
        if abs(fa) < best_abs:
            best_abs = abs(fa)
            best_x = xa
        if abs(fm) < best_abs:
            best_abs = abs(fm)
            best_x = xm
        if abs(fb) < best_abs:
            best_abs = abs(fb)
            best_x = xb

        if abs(fa) <= 1e-8:
            return xa
        if abs(fm) <= 1e-8:
            return xm
        if abs(fb) <= 1e-8:
            return xb

        # IQI: find next point x_new using inverse quadratic interpolation
        # through (xa,fa), (xm,fm), (xb,fb)
        # Formula: x_new = xa*fa*(fm - fb)/((fa - fm)*(fa - fb))
        #          + xm*fm*(fa - fb)/((fm - fa)*(fm - fb))
        #          + xb*fb*(fa - fm)/((fb - fa)*(fb - fm))

        denom1 = (fa - fm) * (fa - fb)
        denom2 = (fm - fa) * (fm - fb)
        denom3 = (fb - fa) * (fb - fm)

        x_new = None
        if abs(denom1) > 1e-300 and abs(denom2) > 1e-300 and abs(denom3) > 1e-300:
            try:
                x_new = (xa * fa * (fm - fb) / denom1
                         + xm * fm * (fa - fb) / denom2
                         + xb * fb * (fa - fm) / denom3)
            except (ZeroDivisionError, OverflowError):
                x_new = None

        # Safeguard: if x_new is outside [min(xa,xb), max(xa,xb)], use bisection
        lo = min(xa, xb)
        hi = max(xa, xb)
        if x_new is None or not math.isfinite(x_new) or x_new <= lo or x_new >= hi:
            x_new = 0.5 * (xa + xb)

        f_new = f(x_new)
        if not math.isfinite(f_new):
            # Fallback: bisection
            x_new = 0.5 * (xa + xb)
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break

        if abs(f_new) < best_abs:
            best_abs = abs(f_new)
            best_x = x_new
        if abs(f_new) <= 1e-8:
            return x_new

        # Update the 3 points: keep the bracket and the best interior point
        # The new bracket is between the two points with opposite signs
        # Replace the point with the largest |f| that is not part of the new bracket

        # Determine new bracket
        if fa * f_new < 0:
            # New bracket: [xa, x_new], drop xb (or keep xm if it's inside)
            # We want to keep 3 points: xa, x_new, and the best of {xm, xb}
            # Actually, we need a valid bracket. Let's set:
            # new xa = xa, new xb = x_new, and xm = the other interior point
            # But we need 3 distinct points for IQI. Let's keep the 3 with
            # the smallest |f| values that maintain a bracket.
            # Simple approach: new bracket is [xa, x_new] (if fa*f_new<0)
            # Keep xm as the third point (it was inside the old bracket)
            # But xm might be outside [xa, x_new]. Let's just re-bisect.
            # Simpler: use the two bracket points + a new bisection point
            # Actually, let's just keep it simple:
            # The bracket is [xa, x_new]. The third point should be inside.
            # If xm is inside [xa, x_new], keep it. Otherwise, bisect.
            if lo <= xm <= hi or min(xa, x_new) <= xm <= max(xa, x_new):
                # xm is inside the new bracket, keep it
                xb, fb = x_new, f_new
            else:
                # Need a new interior point; bisect the new bracket
                xm = 0.5 * (xa + x_new)
                fm = f(xm)
                xb, fb = x_new, f_new
                if not math.isfinite(fm):
                    break
        elif fb * f_new < 0:
            # New bracket: [x_new, xb], drop xa
            if min(x_new, xb) <= xm <= max(x_new, xb):
                xa, fa = x_new, f_new
            else:
                xa, fa = x_new, f_new
                xm = 0.5 * (x_new + xb)
                fm = f(xm)
                if not math.isfinite(fm):
                    break
        else:
            # f_new has same sign as both fa and fb — this shouldn't happen
            # if we started with a valid bracket, but handle gracefully
            # Use bisection
            x_new = 0.5 * (xa + xb)
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
            if abs(f_new) < best_abs:
                best_abs = abs(f_new)
                best_x = x_new
            if fa * f_new < 0:
                xb, fb = x_new, f_new
                xm = 0.5 * (xa + xb)
                fm = f(xm)
                if not math.isfinite(fm):
                    break
            elif fb * f_new < 0:
                xa, fa = x_new, f_new
                xm = 0.5 * (xa + xb)
                fm = f(xm)
                if not math.isfinite(fm):
                    break
            else:
                break

        # Ensure ordering for next IQI: xa and xb are the bracket ends
        # (fa*fb < 0), xm is inside
        if abs(fa) > abs(fb):
            xa, xb = xb, xa
            fa, fb = fb, fa

    return best_x
# EVOLVE-BLOCK-END
