"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f near x0 using spectral-aware bracketing:
    1. Estimate dominant period via frequency sweep
    2. Place dense brackets using estimated period
    3. Bisection + Newton hybrid for convergence
    """
    x0 = float(x0)
    calls = 0
    
    def _f(x):
        nonlocal calls
        calls += 1
        val = f(x)
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    
    # Step 1: Check if x0 is already a root
    fx0 = _f(x0)
    if math.isfinite(fx0) and abs(fx0) <= 1e-8:
        return x0
    
    # Step 2: Estimate dominant period via frequency sweep
    # Try several candidate periods k where spacing = 1/(2*pi*k)
    # We look for the k that gives consistent sign changes (indicating we're near a root)
    
    # First, try a few Newton steps to see if we can converge quickly
    x = x0
    fx = fx0
    newton_success = False
    for _ in range(8):
        if not math.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        h = 1e-7 * max(1.0, abs(x))
        fph = _f(x + h)
        if not math.isfinite(fph):
            break
        d = (fph - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new
        fx = _f(x)
    if math.isfinite(fx) and abs(fx) <= 1e-8:
        return x
    
    # Step 3: Spectral-aware bracketing
    # Try to estimate the period by sampling at different frequencies
    # For oscillatory functions like sin(x), cos(x), etc., the period is ~2*pi
    
    # Candidate periods: assume frequencies k in [0.5, 1, 2, 4, 8] (periods 4pi, 2pi, pi, pi/2, pi/4)
    candidate_ks = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    best_bracket = None
    best_bracket_width = float('inf')
    
    for k in candidate_ks:
        period = 2.0 * math.pi / k
        # Sample points spaced by period/4 (quarter period) to catch sign changes
        spacing = period / 4.0
        
        # Search in a window of about 2 periods around x0
        n_points = int(2 * 4)  # 2 periods * 4 points per period = 8 points each direction
        prev_x = None
        prev_f = None
        
        for i in range(-n_points, n_points + 1):
            xi = x0 + i * spacing
            fi = _f(xi)
            if not math.isfinite(fi):
                prev_x = None
                prev_f = None
                continue
            if abs(fi) <= 1e-8:
                return xi
            if prev_f is not None and math.isfinite(prev_f) and prev_f * fi < 0:
                bracket = (prev_x, xi, prev_f, fi)
                width = abs(xi - prev_x)
                if width < best_bracket_width:
                    best_bracket = bracket
                    best_bracket_width = width
            prev_x = xi
            prev_f = fi
        
        # If we found a bracket, use it
        if best_bracket is not None:
            break
    
    # If no bracket found via spectral sweep, fall back to geometric expansion
    if best_bracket is None:
        step = max(1e-3, abs(x0) * 1e-3)
        for direction in [1, -1]:
            for i in range(20):
                xi = x0 + direction * step * (2 ** i)
                fi = _f(xi)
                if not math.isfinite(fi):
                    continue
                if abs(fi) <= 1e-8:
                    return xi
                if math.isfinite(fx0) and fx0 * fi < 0:
                    if direction == 1:
                        best_bracket = (x0, xi, fx0, fi)
                    else:
                        best_bracket = (xi, x0, fi, fx0)
                    break
            if best_bracket is not None:
                break
    
    # If still no bracket, try a denser linear sweep
    if best_bracket is None:
        base_step = max(1e-4, abs(x0) * 1e-4)
        prev_x = None
        prev_f = None
        for i in range(-100, 101):
            xi = x0 + i * base_step
            fi = _f(xi)
            if not math.isfinite(fi):
                prev_x = None
                prev_f = None
                continue
            if abs(fi) <= 1e-8:
                return xi
            if prev_f is not None and math.isfinite(prev_f) and prev_f * fi < 0:
                best_bracket = (prev_x, xi, prev_f, fi)
                break
            prev_x = xi
            prev_f = fi
    
    # If no bracket found, try final Newton from x0
    if best_bracket is None:
        x = x0
        fx = fx0
        for _ in range(30):
            if not math.isfinite(fx):
                break
            if abs(fx) <= 1e-8:
                return x
            h = 1e-8 * max(1.0, abs(x))
            fph = _f(x + h)
            if not math.isfinite(fph):
                break
            d = (fph - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            x = x_new
            fx = _f(x)
        return x if math.isfinite(x) else x0
    
    # Step 4: Bisection + Newton hybrid to converge within the bracket
    a, b, fa, fb = best_bracket
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    lo, hi = a, b
    flo, fhi = fa, fb
    
    for _ in range(100):
        if not math.isfinite(flo) or not math.isfinite(fhi):
            break
        if abs(flo) <= 1e-8:
            return lo
        if abs(fhi) <= 1e-8:
            return hi
        
        mid = (lo + hi) / 2.0
        fmid = _f(mid)
        if not math.isfinite(fmid):
            if flo * fmid < 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
            continue
        
        if abs(fmid) <= 1e-8:
            return mid
        
        # Try Newton step
        h = 1e-8 * max(1.0, abs(mid))
        fmid_h = _f(mid + h)
        if math.isfinite(fmid_h):
            d = (fmid_h - fmid) / h
            if d != 0 and math.isfinite(d):
                x_new = mid - fmid / d
                if math.isfinite(x_new) and lo < x_new < hi:
                    fnew = _f(x_new)
                    if math.isfinite(fnew):
                        if abs(fnew) <= 1e-8:
                            return x_new
                        if flo * fnew < 0:
                            hi = x_new
                            fhi = fnew
                        else:
                            lo = x_new
                            flo = fnew
                        continue
        
        # Bisection step
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
        
        if abs(hi - lo) < 1e-12 * max(1.0, abs(mid)):
            break
    
    # Return the best estimate
    candidates = [lo, hi, (lo + hi) / 2.0]
    best_x = lo
    best_f = abs(flo) if math.isfinite(flo) else float('inf')
    for c in candidates:
        if not math.isfinite(c):
            continue
        fc = _f(c)
        if math.isfinite(fc) and abs(fc) < best_f:
            best_f = abs(fc)
            best_x = c
    
    return best_x
# EVOLVE-BLOCK-END
