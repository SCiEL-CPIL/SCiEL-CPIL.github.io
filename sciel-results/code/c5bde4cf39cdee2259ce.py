"""Root finding for ill-conditioned equations: Global + Local Hybrid."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin below 150s kill
    MAX_EVALS = 490   # Safety margin below 500 limit

    eval_count = 0
    best_x = float(x0)
    best_f = None

    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS:
            return True
        if time.time() - start_time > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f
        if check_budget():
            return None
        try:
            val = f(x)
            eval_count += 1
            if val is not None and np.isfinite(val):
                if best_f is None or abs(val) < abs(best_f):
                    best_f = val
                    best_x = x
                return val
        except Exception:
            pass
        return None

    def update_best(x, fx):
        nonlocal best_x, best_f
        if fx is not None and np.isfinite(fx):
            if best_f is None or abs(fx) < abs(best_f):
                best_f = fx
                best_x = x

    def g(x):
        """Surrogate objective: f(x)^2, minimized to find root."""
        fx = safe_eval(x)
        if fx is None or not np.isfinite(fx):
            return float('inf')
        return fx * fx

    # First evaluate at x0
    fx0 = safe_eval(x0)
    if fx0 is None:
        return best_x
    update_best(x0, fx0)
    if abs(fx0) <= 1e-8:
        return best_x

    # Phase 1: Global search via Nelder-Mead on g(x) = f(x)^2
    # This explores broadly without derivative instability
    x0_val = float(x0)
    scale = max(1.0, abs(x0_val)) * 1e-2
    
    # Initial simplex vertices for 1D Nelder-Mead
    # In 1D, Nelder-Mead reduces to a simple adaptive search
    # We'll use a multi-start golden section + local refinement approach
    
    # Golden section search over expanding intervals
    def golden_section_search(lo, hi, max_iter=30):
        """Minimize g(x) on [lo, hi] using golden section method."""
        if check_budget():
            return best_x, best_f
        gr = (1 + math.sqrt(5)) / 2
        invgr = 1 / gr
        invgr2 = 1 / (gr * gr)
        
        c = hi - (hi - lo) * invgr
        d = lo + (hi - lo) * invgr
        
        fc = g(c)
        fd = g(d)
        
        if fc < abs(best_f) if best_f is not None else True:
            update_best(c, safe_eval(c) if fc != float('inf') else None)
        if fd < abs(best_f) if best_f is not None else True:
            update_best(d, safe_eval(d) if fd != float('inf') else None)
        
        for _ in range(max_iter):
            if check_budget():
                break
            if fc <= fd:
                hi = d
                d = c
                fd = fc
                c = hi - (hi - lo) * invgr
                fc = g(c)
            else:
                lo = c
                c = d
                fc = fd
                d = lo + (hi - lo) * invgr
                fd = g(d)
            if hi - lo < 1e-12 * max(1.0, abs(lo)):
                break
        
        mid = (lo + hi) / 2.0
        fmid = safe_eval(mid)
        if fmid is not None and np.isfinite(fmid):
            update_best(mid, fmid)
            return mid, fmid
        return best_x, best_f

    # Multi-scale golden section search
    scales = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for s in scales:
        if check_budget():
            break
        interval = max(scale * s, 1e-10)
        # Search in both directions
        lo1 = x0_val - interval
        hi1 = x0_val + interval
        golden_section_search(lo1, hi1, max_iter=20)
        
        # Also try asymmetric intervals
        lo2 = x0_val - interval * 3
        hi2 = x0_val + interval
        golden_section_search(lo2, hi2, max_iter=15)
        
        lo3 = x0_val - interval
        hi3 = x0_val + interval * 3
        golden_section_search(lo3, hi3, max_iter=15)
        
        if best_f is not None and abs(best_f) <= 1e-8:
            return best_x

    # Phase 2: If we have a good point, try to bracket around it
    if best_f is not None:
        a = best_x
        fa = best_f
        b = None
        fb = None
        
        step = max(1.0, abs(a)) * 1e-4
        for i in range(60):
            if check_budget():
                break
            x_try = a + step
            f_try = safe_eval(x_try)
            if f_try is not None and np.isfinite(f_try):
                update_best(x_try, f_try)
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    break
                elif abs(f_try) < abs(fa):
                    a = x_try
                    fa = f_try
                    step *= 2
                else:
                    step *= 2
            
            x_try = a - step
            f_try = safe_eval(x_try)
            if f_try is not None and np.isfinite(f_try):
                update_best(x_try, f_try)
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    break
                elif abs(f_try) < abs(fa):
                    a = x_try
                    fa = f_try
                    step *= 2
                else:
                    step *= 2
            
            if step > 1e12:
                break

        # Phase 3: If bracketed, use Brent's method (bisection + secant + inverse quadratic)
        if b is not None and fb is not None and fa * fb < 0:
            lo, hi = min(a, b), max(a, b)
            flo, fhi = fa, fb
            if abs(flo) > abs(fhi):
                lo, hi = hi, lo
                flo, fhi = fhi, flo

            x = lo
            fx = flo
            prev_x = x + (hi - lo) * 0.5
            prev_fx = None
            prev2_x = x
            prev2_fx = None

            for _ in range(100):
                if check_budget():
                    break
                if abs(fx) <= 1e-8:
                    return x

                # Try inverse quadratic interpolation if we have 3 points
                x_new = None
                if prev_fx is not None and prev2_fx is not None:
                    x1, x2, x3 = x, prev_x, prev2_x
                    f1, f2, f3 = fx, prev_fx, prev2_fx
                    # Lagrange interpolation for root
                    denom = f1*(f2-f3) + f2*(f3-f1) + f3*(f1-f2)
                    if abs(denom) > 1e-300:
                        num = x1*f1*(f2-f3) + x2*f2*(f3-f1) + x3*f3*(f1-f2)
                        x_new = num / denom
                
                # Try secant if we have 2 points
                if x_new is None and prev_fx is not None:
                    if fx != prev_fx:
                        x_new = x - fx * (x - prev_x) / (fx - prev_fx)

                # Bisection fallback
                if x_new is None or not np.isfinite(x_new) or x_new <= lo or x_new >= hi:
                    x_new = (lo + hi) / 2.0

                f_new = safe_eval(x_new)
                if f_new is None or not np.isfinite(f_new):
                    break
                update_best(x_new, f_new)

                if f_new * flo < 0:
                    hi = x_new
                    fhi = f_new
                else:
                    lo = x_new
                    flo = f_new

                # Track history for interpolation
                prev2_x = prev_x if prev_fx is not None else x
                prev2_fx = prev_fx
                prev_x = x
                prev_fx = fx
                x = x_new
                fx = f_new

            return best_x

    # Phase 4: No bracket - use damped secant/Newton from best point
    x1 = best_x if best_x is not None else float(x0)
    fx1 = best_f if best_f is not None else fx0
    if fx1 is None:
        return best_x

    x2 = x1 + max(1.0, abs(x1)) * 1e-4
    fx2 = safe_eval(x2)
    if fx2 is None or not np.isfinite(fx2):
        return best_x
    update_best(x2, fx2)

    for _ in range(120):
        if check_budget():
            break
        if abs(fx1) <= 1e-8:
            return x1

        # Secant step
        if fx2 != fx1 and abs(fx1 - fx2) > 1e-300:
            x_new = x1 - fx1 * (x1 - x2) / (fx1 - fx2)
        else:
            # Newton with finite difference
            h = 1e-6 * max(1.0, abs(x1))
            f_xh = safe_eval(x1 + h)
            if f_xh is None or not np.isfinite(f_xh):
                break
            d = (f_xh - fx1) / h
            if d == 0 or not np.isfinite(d):
                break
            x_new = x1 - fx1 / d

        if not np.isfinite(x_new):
            break

        # Damping
        step_size = abs(x_new - x1)
        max_step = max(1.0, abs(x1)) * 10.0
        if step_size > max_step:
            x_new = x1 + math.copysign(max_step, x_new - x1)

        f_new = safe_eval(x_new)
        if f_new is None or not np.isfinite(f_new):
            break
        update_best(x_new, f_new)

        x2 = x1
        fx2 = fx1
        x1 = x_new
        fx1 = f_new

    return best_x

# EVOLVE-BLOCK-END
