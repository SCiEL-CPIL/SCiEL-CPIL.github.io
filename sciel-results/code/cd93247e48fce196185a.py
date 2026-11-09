"""Root finder for 3 fixed oscillatory equations with optimized call budget."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x with |f(x)| <= 1e-8.
    
    Strategy:
    1. Adaptive geometric bracketing search.
    2. Illinois method for fast convergence.
    3. Polish pass with tight tolerance.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    tol = 1e-8
    
    x0 = float(x0)
    
    try:
        f0 = f(x0)
    except Exception:
        x_try = x0 + 1e-6
        try:
            f0 = f(x_try)
            x0 = x_try
        except Exception:
            return x0
    
    if not math.isfinite(f0):
        return x0
        
    if abs(f0) <= tol:
        return x0

    # Adaptive geometric bracketing
    d = 1e-4
    max_d = 100.0
    growth = 2.0  # Geometric growth factor
    
    a = b = fa = fb = None
    best_x, best_f, best_abs = x0, f0, abs(f0)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if math.isfinite(x) and math.isfinite(fx):
            af = abs(fx)
            if af < best_abs:
                best_x, best_f, best_abs = x, fx, af

    # Try both directions with geometric expansion
    while d <= max_d and time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Check left side first
        x_left = x0 - d
        try:
            fl = f(x_left)
        except Exception:
            fl = float('nan')
            
        if math.isfinite(fl):
            update_best(x_left, fl)
            if abs(fl) <= tol:
                return x_left
            if fa is None:
                fa, a = fl, x_left
            if fa * f0 < 0:
                a, fa, b, fb = x_left, fl, x0, f0
                break
        
        # Check right side
        x_right = x0 + d
        try:
            fr = f(x_right)
        except Exception:
            fr = float('nan')
            
        if math.isfinite(fr):
            update_best(x_right, fr)
            if abs(fr) <= tol:
                return x_right
            if fb is None:
                fb, b = fr, x_right
            if f0 * fr < 0:
                a, fa, b, fb = x0, f0, x_right, fr
                break
        
        d *= growth

    # Illinois method for bracketed root finding
    if a is not None and b is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Illinois iteration
        for _ in range(60):
            if time.time() > deadline:
                break
                
            if abs(fa) <= tol: return a
            if abs(fb) <= tol: return b
            
            # Illinois: if |fa| > |fb|, shrink a side by halving fa
            if abs(fa) > abs(fb):
                a = (a + b) / 2
                fa = (fa + fb) / 2
            
            # Secant step
            denom = fb - fa
            if abs(denom) < 1e-300:
                x_new = (a + b) / 2
            else:
                x_new = b - fb * (b - a) / denom
            
            if not (a < x_new < b):
                x_new = (a + b) / 2
                
            if not math.isfinite(x_new):
                break
                
            try:
                f_new = f(x_new)
            except Exception:
                break
                
            if not math.isfinite(f_new):
                break
                
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
                
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
                
            if abs(b - a) < 1e-14 * max(1.0, abs(b)):
                break
        
        # Polish with tighter tolerance
        if best_abs <= 1e-7:
            if a is not None and b is not None and min(a,b) <= best_x <= max(a,b):
                pa, pb, pfa, pfb = a, b, fa, fb
            else:
                eps = 1e-12
                pa, pb = best_x - eps, best_x + eps
                try:
                    pfa, pfb = f(pa), f(pb)
                except Exception:
                    pfa = pfb = float('nan')
                
            if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                for _ in range(15):
                    if time.time() > deadline:
                        break
                    if abs(pfa) <= tol: return pa
                    if abs(pfb) <= tol: return pb
                    
                    x_mid = (pa + pb) / 2
                    try:
                        f_mid = f(x_mid)
                    except Exception:
                        break
                    if not math.isfinite(f_mid):
                        break
                    update_best(x_mid, f_mid)
                    if abs(f_mid) <= tol:
                        return x_mid
                    
                    if pfa * f_mid < 0:
                        pb, pfb = x_mid, f_mid
                    else:
                        pa, pfa = x_mid, f_mid

        return best_x if best_abs <= tol else (a + b) / 2

    # No bracket: use secant from best points
    else:
        x1, f1 = x0, f0
        x2, f2 = best_x, best_f
        
        if abs(x2 - x1) < 1e-15:
            x2 = x1 + 1e-6
            try:
                f2 = f(x2)
            except Exception:
                return best_x
            if not math.isfinite(f2):
                return best_x
            update_best(x2, f2)

        for _ in range(40):
            if time.time() > deadline:
                break
                
            if abs(f1) <= tol: return x1
            if abs(f2) <= tol: return x2
            
            denom = f2 - f1
            if abs(denom) < 1e-300:
                break
                
            x_new = x2 - f2 * (x2 - x1) / denom
            
            if not math.isfinite(x_new):
                break
                
            try:
                f_new = f(x_new)
            except Exception:
                break
                
            if not math.isfinite(f_new):
                break
                
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
                
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
            
        # Final polish
        if best_abs <= 1e-7:
            eps = 1e-12
            pa, pb = best_x - eps, best_x + eps
            try:
                pfa, pfb = f(pa), f(pb)
            except Exception:
                return best_x
            
            if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                for _ in range(15):
                    if time.time() > deadline:
                        break
                    if abs(pfa) <= tol: return pa
                    if abs(pfb) <= tol: return pb
                    
                    x_mid = (pa + pb) / 2
                    try:
                        f_mid = f(x_mid)
                    except Exception:
                        break
                    if not math.isfinite(f_mid):
                        break
                    update_best(x_mid, f_mid)
                    if abs(f_mid) <= tol:
                        return x_mid
                    
                    if pfa * f_mid < 0:
                        pb, pfb = x_mid, f_mid
                    else:
                        pa, pfa = x_mid, f_mid

        return best_x

# EVOLVE-BLOCK-END
