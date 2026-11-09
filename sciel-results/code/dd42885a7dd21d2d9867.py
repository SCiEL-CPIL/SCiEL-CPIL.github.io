"""Root finding for ill-conditioned equations: secant + manual Brent + aggressive polish."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def find_root(f, x0):
    t_start = time.time()
    deadline = t_start + 135.0
    max_evals = 500
    eval_count = [0]
    
    def safe_eval(x):
        if eval_count[0] >= max_evals:
            raise RuntimeError("Eval budget exceeded")
        if time.time() > deadline:
            raise RuntimeError("Time deadline exceeded")
        eval_count[0] += 1
        return f(x)
    
    def check_time():
        if time.time() > deadline:
            raise RuntimeError("Time deadline exceeded")
    
    def manual_brent(fa, a, fb, b, xtol=1e-14, rtol=1e-14, maxiter=100):
        """Manual Brent's method using safe_eval for accounting."""
        if (fa > 0) == (fb > 0):
            return None, None
        # Ensure |fa| < |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        c = a
        fc = fa
        d = b - a
        e = d
        for _ in range(maxiter):
            check_time()
            if abs(fa) <= 1e-14:
                return a, fa
            if (b - a) < xtol + rtol * abs(a):
                return a, fa
            # Inverse quadratic interpolation
            if abs(fc) > abs(fa):
                c = b
                fc = fb
                d = b - a
                e = d
            m = (a + b) / 2.0
            s = fa / fc
            if abs(a - c) < xtol + rtol * abs(c):
                # Linear interpolation
                p = 2.0 * m * fa
            else:
                p = fa * fa * fc * (a - c) ** 2 - fa * fa * fb * (a - b) ** 2
                q = (fc - fa) * (a - c) ** 2 - (fb - fa) * (a - b) * (a - c)
                if abs(p) > abs(q):
                    p = -p
                    q = -q
                r = p / q
                # Check if r is acceptable
                if ((a < r and r < b) or (b < r and r < a)):
                    if r > m:
                        r = b
                    else:
                        r = a
                    if abs(r - a) >= xtol + rtol * abs(a):
                        if abs(r - b) >= xtol + rtol * abs(b):
                            if abs(r - c) >= xtol:
                                e = d
                                d = r - a
                                a = r
                                try:
                                    fa = safe_eval(a)
                                except RuntimeError:
                                    return a, fa
                                continue
            # Bisection
            d = (b - a) / 2.0
            e = d
            a = a + d
            try:
                fa = safe_eval(a)
            except RuntimeError:
                return a, fa
            if (fa > 0) == (fb > 0):
                c = b
                fc = fb
                b = a
                fb = fa
            else:
                c = b
                fc = fb
                b = a
                fb = fa
        return a, fa
    
    x = float(x0)
    best_x = x
    best_f = safe_eval(x)
    best_abs = abs(best_f)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Secant-based iteration with bracket tracking
    a = x
    fa = best_f
    b = None
    fb = None
    
    x_prev = None
    f_prev = None
    
    # Initial secant step: perturb x0
    h_init = 1e-4 * max(1.0, abs(x))
    x_try = x + h_init
    f_try = safe_eval(x_try)
    
    # Set up secant pair
    if abs(f_try) < best_abs:
        x_prev = x
        f_prev = best_f
        best_x = x_try
        best_f = f_try
        best_abs = abs(f_try)
        x = x_try
    else:
        x_prev = x_try
        f_prev = f_try
        x = x
    
    # Update bracket
    if (fa > 0) != (f_try > 0):
        if abs(fa) < abs(f_try):
            a = x
            fa = best_f
            b = x_try
            fb = f_try
        else:
            a = x_try
            fa = f_try
            b = x
            fb = best_f
    
    for i in range(150):
        check_time()
        
        if best_abs <= 1e-8:
            return best_x
        
        # Secant step
        if x_prev is not None and abs(x - x_prev) > 1e-15:
            d_sec = (best_f - f_prev) / (x - x_prev)
            if d_sec != 0 and np.isfinite(d_sec):
                step = best_f / d_sec
                x_new = x - step
            else:
                # Fallback: small perturbation
                x_new = x + 1e-4 * max(1.0, abs(x))
        else:
            x_new = x + 1e-4 * max(1.0, abs(x))
        
        # Clip step
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + math.copysign(10 * max(1.0, abs(x)), x_new - x)
        
        if not np.isfinite(x_new):
            break
        
        fx_new = safe_eval(x_new)
        abs_new = abs(fx_new)
        
        # Update bracket
        if a is not None and b is not None and fa is not None and fb is not None:
            if (fa > 0) != (fx_new > 0):
                b = x_new
                fb = fx_new
            elif (fb > 0) != (fx_new > 0):
                a = x_new
                fa = fx_new
            elif abs_new < abs(fa):
                a = x_new
                fa = fx_new
            elif abs_new < abs(fb):
                b = x_new
                fb = fx_new
        elif a is not None and fa is not None:
            if (fa > 0) != (fx_new > 0):
                b = x_new
                fb = fx_new
            elif abs_new < abs(fa):
                a = x_new
                fa = fx_new
        else:
            a = x
            fa = best_f
            if (fa > 0) != (fx_new > 0):
                b = x_new
                fb = fx_new
        
        if abs_new < best_abs:
            x_prev = x
            f_prev = best_f
            best_x = x_new
            best_f = fx_new
            best_abs = abs_new
            x = x_new
        else:
            # Secant didn't improve; try bisection if bracket exists
            if a is not None and b is not None and fa is not None and fb is not None:
                if (fa > 0) != (fb > 0):
                    mid = (a + b) / 2.0
                    fm = safe_eval(mid)
                    abs_m = abs(fm)
                    if abs_m < best_abs:
                        best_x = mid
                        best_f = fm
                        best_abs = abs_m
                    if (fa > 0) == (fm > 0):
                        a = mid
                        fa = fm
                    else:
                        b = mid
                        fb = fm
                    x_prev = x
                    f_prev = best_f
                    x = mid
                else:
                    # Reset bracket, try damped secant
                    a = None
                    b = None
                    x_new_damp = x + 0.5 * (x_new - x)
                    if not np.isfinite(x_new_damp):
                        break
                    fx_damp = safe_eval(x_new_damp)
                    if abs(fx_damp) < best_abs:
                        x_prev = x
                        f_prev = best_f
                        best_x = x_new_damp
                        best_f = fx_damp
                        best_abs = abs(fx_damp)
                        x = x_new_damp
                    else:
                        x_prev = x_new
                        f_prev = fx_new
                        x = x_new_damp
            else:
                # No bracket; damped step
                x_new_damp = x + 0.5 * (x_new - x)
                if not np.isfinite(x_new_damp):
                    break
                fx_damp = safe_eval(x_new_damp)
                if abs(fx_damp) < best_abs:
                    x_prev = x
                    f_prev = best_f
                    best_x = x_new_damp
                    best_f = fx_damp
                    best_abs = abs(fx_damp)
                    x = x_new_damp
                else:
                    x_prev = x_new
                    f_prev = fx_new
                    x = x_new_damp
    
    # Phase 2: Manual Brent if bracket exists
    if a is not None and b is not None and fa is not None and fb is not None:
        if (fa > 0) != (fb > 0):
            try:
                if time.time() < deadline and eval_count[0] < max_evals - 10:
                    x_brent, fx_brent = manual_brent(fa, a, fb, b, xtol=1e-14, rtol=1e-14, maxiter=80)
                    if x_brent is not None and fx_brent is not None:
                        if abs(fx_brent) < best_abs:
                            best_x = x_brent
                            best_f = fx_brent
                            best_abs = abs(fx_brent)
            except Exception:
                pass
    
    # Phase 3: Aggressive secant polish (1 eval per iteration)
    if best_abs > 1e-8:
        x = best_x
        fx = best_f
        # Use the best previous point for secant
        px = x_prev if x_prev is not None else x - 1e-6 * max(1.0, abs(x))
        pf = f_prev if f_prev is not None else safe_eval(px)
        
        for j in range(100):
            check_time()
            if abs(fx) <= 1e-14:
                return x
            if eval_count[0] >= max_evals - 1:
                break
            
            if abs(x - px) > 1e-15:
                d_sec = (fx - pf) / (x - px)
                if d_sec != 0 and np.isfinite(d_sec):
                    step = fx / d_sec
                    x_new = x - step
                else:
                    break
            else:
                break
            
            if not np.isfinite(x_new):
                break
            
            fx_new = safe_eval(x_new)
            abs_new = abs(fx_new)
            
            if abs_new < abs(fx):
                px = x
                pf = fx
                x = x_new
                fx = fx_new
                if abs_new < best_abs:
                    best_x = x
                    best_f = fx
                    best_abs = abs_new
                if abs_new <= 1e-8:
                    return x
            else:
                # Damp the step
                x_new_damp = x + 0.5 * (x_new - x)
                if not np.isfinite(x_new_damp):
                    break
                fx_damp = safe_eval(x_new_damp)
                if abs(fx_damp) < abs(fx):
                    px = x
                    pf = fx
                    x = x_new_damp
                    fx = fx_damp
                    if abs(fx) < best_abs:
                        best_x = x
                        best_f = fx
                        best_abs = abs(fx)
                    if abs(fx) <= 1e-8:
                        return x
                else:
                    px = x_new
                    pf = fx_new
                    x = x_new_damp
                    fx = fx_damp
    
    return best_x
# EVOLVE-BLOCK-END
