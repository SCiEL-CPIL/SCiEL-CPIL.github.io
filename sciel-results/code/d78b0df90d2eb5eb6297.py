"""Root finder for 3 fixed smooth polynomial equations; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def _is_finite(x):
    return x is not None and math.isfinite(x)

def _safe_f(f, x):
    try:
        y = f(x)
    except Exception:
        return None
    if not _is_finite(y):
        return None
    return float(y)

def _secant_step(a, fa, b, fb):
    denom = fb - fa
    if denom == 0.0:
        return (a + b) / 2.0
    s = b - (fb * (b - a)) / denom
    return s

def find_root(f, x0):
    start = time.time()
    deadline = start + 110.0
    MAX_CALLS = 400
    calls = 0
    TOL = 1e-12

    def g(x):
        nonlocal calls
        if calls >= MAX_CALLS:
            return None
        y = _safe_f(f, x)
        calls += 1
        return y

    best_x = None
    best_f = None
    best_abs = math.inf

    def consider(x, fx):
        nonlocal best_x, best_f, best_abs
        if fx is None or not _is_finite(x):
            return
        a = abs(fx)
        if a < best_abs:
            best_abs = a
            best_x = x
            best_f = fx

    x = float(x0)
    fx = g(x)
    consider(x, fx)
    if fx is not None and abs(fx) <= TOL:
        return x

    # Bracket search: try nearby points first, then expand
    a = b = None
    fa = fb = None
    if fx is not None:
        step = max(1.0, abs(x)) * 0.5
        for trial in (x + step, x - step):
            ft = g(trial)
            consider(trial, ft)
            if ft is None:
                continue
            if abs(ft) <= TOL:
                return trial
            if fx * ft < 0.0:
                if fx < 0:
                    a, b = x, trial
                    fa, fb = fx, ft
                else:
                    a, b = trial, x
                    fa, fb = ft, fx
                break
        if a is None:
            base = max(1.0, abs(x))
            for mult in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0):
                d = base * mult
                for sign in (1.0, -1.0):
                    trial = x + sign * d
                    ft = g(trial)
                    consider(trial, ft)
                    if ft is None:
                        continue
                    if abs(ft) <= TOL:
                        return trial
                    if fx * ft < 0.0:
                        if fx < 0:
                            a, b = x, trial
                            fa, fb = fx, ft
                        else:
                            a, b = trial, x
                            fa, fb = ft, fx
                        break
                if a is not None:
                    break

    # Main loop: safeguarded secant with bisection fallback
    if a is not None and b is not None:
        for _ in range(300):
            if time.time() > deadline:
                break
            if calls >= MAX_CALLS:
                break
            if abs(fb) <= TOL:
                return b
            if abs(fa) <= TOL:
                return a
            
            # Try secant step
            s = _secant_step(a, fa, b, fb)
            lo, hi = (a, b) if a < b else (b, a)
            
            # Safeguard: must be inside bracket
            if not (lo < s < hi):
                s = (a + b) / 2.0
            
            fs = g(s)
            consider(s, fs)
            if fs is None:
                break
            if abs(fs) <= TOL:
                return s
            
            # Update bracket
            if fa * fs < 0.0:
                b, fb = s, fs
            elif fb * fs < 0.0:
                a, fa = s, fs
            else:
                # No sign change: keep the side with smaller |f|
                if abs(fs) < abs(fa):
                    a, fa = s, fs
                else:
                    b, fb = s, fs
        
        # Return best bracket endpoint
        if abs(fa) <= abs(fb):
            return a
        return b

    # No bracket found: use secant method with damping
    xa = float(x0)
    fa = g(xa)
    consider(xa, fa)
    if fa is not None and abs(fa) <= TOL:
        return xa
    
    h = max(1e-4, abs(xa) * 1e-4)
    xb = xa + h
    fb = g(xb)
    consider(xb, fb)
    if fb is not None and abs(fb) <= TOL:
        return xb

    ref_x = best_x
    ref_f = best_f
    for _ in range(300):
        if time.time() > deadline:
            break
        if calls >= MAX_CALLS:
            break
        if fb is None or fa is None:
            break
        if abs(fb) <= TOL:
            return xb
        if abs(fa) <= TOL:
            return xa
        
        s = _secant_step(xa, fa, xb, fb)
        if not _is_finite(s):
            s = (xa + xb) / 2.0
        
        # Damp if step is too large
        if ref_x is not None:
            max_step = max(1.0, abs(ref_x)) * 1e6
            if abs(s - ref_x) > max_step:
                s = ref_x + math.copysign(max_step, s - ref_x)
        
        fs = g(s)
        consider(s, fs)
        if fs is None:
            break
        if abs(fs) <= TOL:
            return s
        
        # Standard secant update: replace older point
        if abs(fs) < abs(fa):
            xa, fa = xb, fb
            xb, fb = s, fs
        else:
            xa, fa = s, fs
        
        if abs(fs) < abs(ref_f if ref_f is not None else math.inf):
            ref_x, ref_f = s, fs

    # Final polish on best point
    if best_x is not None:
        px = best_x
        pf = best_f
        for _ in range(8):
            if time.time() > deadline:
                break
            if calls >= MAX_CALLS:
                break
            if abs(pf) <= 1e-14:
                return px
            hh = 1e-6 * max(1.0, abs(px))
            qx = px + hh
            qf = g(qx)
            consider(qx, qf)
            if qf is None:
                break
            denom = qf - pf
            if denom == 0.0:
                break
            nx = px - pf * (qx - px) / denom
            if not _is_finite(nx):
                break
            nf = g(nx)
            consider(nx, nf)
            if nf is None:
                break
            if abs(nf) <= 1e-14:
                return nx
            px, pf = nx, nf

    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
