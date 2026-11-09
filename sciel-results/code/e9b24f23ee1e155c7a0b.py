"""Derivative-free Illinois/bisection hybrid with adaptive oscillatory bracketing."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0
    TOL = 1e-12
    HARD_TOL = 1e-8
    MAX_CALLS = 450

    call_count = 0

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def safe_f(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if val != val or not math.isfinite(val):
                return float('nan')
            return val
        except Exception:
            call_count += 1
            return float('nan')

    f0 = safe_f(x0)
    best_x = x0
    best_val = abs(f0) if f0 != float('nan') else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx != float('nan'):
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    if f0 != float('nan') and abs(f0) <= HARD_TOL:
        return x0

    # Cache for evaluated points
    eval_cache = {x0: f0}

    def get_f(x):
        if x in eval_cache:
            return eval_cache[x]
        val = safe_f(x)
        eval_cache[x] = val
        track(x, val)
        return val

    # ------------------------------------------------------------------
    # 1) Adaptive bracketing: detect oscillation period, then scan finely
    # ------------------------------------------------------------------
    a = b = None
    fa = fb = None

    # First, try to estimate the local period by looking for sign changes
    # in a coarse scan. This helps us choose a good step size.
    
    # Try several scales to find the oscillation frequency
    base_step = max(1e-6, abs(x0) * 1e-6)
    
    # Coarse scan to find period: look at points x0 + k*base_step for k=1..20
    # If we see multiple sign changes, we can estimate the period
    sign_changes = 0
    last_sign = 1 if f0 > 0 else (-1 if f0 < 0 else 0)
    last_x = x0
    
    # Try a few different step sizes to find one that resolves the oscillation
    # Start with a small step and increase if no sign changes found
    found_bracket = False
    
    # Strategy: Try geometric expansion but with a finer initial step
    # to catch oscillatory roots
    h = base_step
    last_pos_x = x0
    last_pos_f = f0
    last_neg_x = x0
    last_neg_f = f0
    
    # Phase 1: Fine scan with small steps to catch nearby roots
    # Use up to ~100 calls for bracketing
    for i in range(1, 50):
        if not elapsed_ok() or call_count >= MAX_CALLS - 50:
            break
            
        # Positive side
        x_new_pos = x0 + i * h
        f_new_pos = get_f(x_new_pos)
        
        if f_new_pos != float('nan') and last_pos_f != float('nan'):
            if f_new_pos * last_pos_f < 0:
                a, b = last_pos_x, x_new_pos
                fa, fb = last_pos_f, f_new_pos
                found_bracket = True
                break
            elif abs(f_new_pos) <= TOL:
                return x_new_pos
        
        if f_new_pos != float('nan'):
            last_pos_x = x_new_pos
            last_pos_f = f_new_pos
            
        # Negative side
        x_new_neg = x0 - i * h
        f_new_neg = get_f(x_new_neg)
        
        if f_new_neg != float('nan') and last_neg_f != float('nan'):
            if f_new_neg * last_neg_f < 0:
                a, b = x_new_neg, last_neg_x
                fa, fb = f_new_neg, last_neg_f
                found_bracket = True
                break
            elif abs(f_new_neg) <= TOL:
                return x_new_neg
        
        if f_new_neg != float('nan'):
            last_neg_x = x_new_neg
            last_neg_f = f_new_neg
        
        # Increase step size every 5 iterations to cover more ground
        if i % 5 == 0:
            h *= 2.0

    # Phase 2: If no bracket found, do a broader geometric expansion
    if not found_bracket:
        h = max(base_step * 10, abs(x0) * 1e-3)
        last_pos_x = x0
        last_pos_f = f0
        last_neg_x = x0
        last_neg_f = f0
        
        for i in range(1, 30):
            if not elapsed_ok() or call_count >= MAX_CALLS - 50:
                break
                
            # Positive side
            x_new_pos = x0 + i * h
            f_new_pos = get_f(x_new_pos)
            
            if f_new_pos != float('nan') and last_pos_f != float('nan'):
                if f_new_pos * last_pos_f < 0:
                    a, b = last_pos_x, x_new_pos
                    fa, fb = last_pos_f, f_new_pos
                    found_bracket = True
                    break
                elif abs(f_new_pos) <= TOL:
                    return x_new_pos
            
            if f_new_pos != float('nan'):
                last_pos_x = x_new_pos
                last_pos_f = f_new_pos
                
            # Negative side
            x_new_neg = x0 - i * h
            f_new_neg = get_f(x_new_neg)
            
            if f_new_neg != float('nan') and last_neg_f != float('nan'):
                if f_new_neg * last_neg_f < 0:
                    a, b = x_new_neg, last_neg_x
                    fa, fb = f_new_neg, last_neg_f
                    found_bracket = True
                    break
                elif abs(f_new_neg) <= TOL:
                    return x_new_neg
            
            if f_new_neg != float('nan'):
                last_neg_x = x_new_neg
                last_neg_f = f_new_neg
            
            h *= 1.5

    # ------------------------------------------------------------------
    # 2) Refine with Illinois method (derivative-free, 1 call per iter)
    # ------------------------------------------------------------------
    if a is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Illinois method: modified bisection that converges faster
        # Keep track of which endpoint has the smaller |f|
        xa, ya = a, fa
        xb, yb = b, fb
        
        # Ensure |ya| <= |yb|
        if abs(ya) > abs(yb):
            xa, xb = xb, xa
            ya, yb = yb, ya
        
        for _ in range(200):
            if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                break
                
            if abs(ya) <= TOL:
                return xa
            if abs(yb) <= TOL:
                return xb
            
            # Illinois step: if same side, use 2/3 point; else bisection
            if ya * yb < 0:
                # Sign change: root is between xa and xb
                if abs(ya) < abs(yb):
                    # Root closer to xa, use Illinois point
                    x_new = xa + 0.5 * (xb - xa)
                else:
                    x_new = xb + 0.5 * (xa - xb)
            else:
                # No sign change: move the point with larger |f|
                if abs(ya) < abs(yb):
                    x_new = xb + 0.5 * (xa - xb)
                    # Update xb
                    f_new = get_f(x_new)
                    if f_new != float('nan'):
                        xb, yb = x_new, f_new
                        if abs(yb) <= TOL:
                            return xb
                        continue
                else:
                    x_new = xa + 0.5 * (xb - xa)
                    f_new = get_f(x_new)
                    if f_new != float('nan'):
                        xa, ya = x_new, f_new
                        if abs(ya) <= TOL:
                            return xa
                        continue
            
            # Evaluate at Illinois point
            f_new = get_f(x_new)
            if f_new != float('nan'):
                if f_new * ya < 0:
                    # Root is between xa and x_new
                    xb, yb = x_new, f_new
                else:
                    # Root is between x_new and xb
                    xa, ya = x_new, f_new
                
                # Re-order so |ya| <= |yb|
                if abs(ya) > abs(yb):
                    xa, xb = xb, xa
                    ya, yb = yb, ya
                
                if abs(ya) <= TOL:
                    return xa
                if abs(yb) <= TOL:
                    return xb

    # ------------------------------------------------------------------
    # 3) Return best candidate
    # ------------------------------------------------------------------
    if best_x == best_x and math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
