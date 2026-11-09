import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using Global Grid Search + Brent's Method.
    
    Strategy:
    1. Assume the oscillatory functions have roots within a moderate range around x0.
    2. Perform a coarse grid search to identify intervals where f(x) changes sign.
       This is robust to the specific shape of the oscillation as long as the 
       step size is smaller than the period.
    3. Once a bracket [a, b] with a sign change is found, use Brent's method 
       (implemented manually to avoid SciPy dependency issues in some environments, 
       or use SciPy if available and safe) to converge to the root.
       Brent's method combines bisection, secant, and inverse quadratic interpolation, 
       offering superlinear convergence and robustness.
    4. If no sign change is found in the grid, fallback to minimizing |f(x)| locally 
       using a simple ternary search or golden section search, as the root might be 
       tangential (double root) or the grid missed a narrow interval.
    
    This approach is structurally different from local minimization of f(x)^2 because 
    it explicitly seeks sign changes (global search) before refining, which is often 
    faster for oscillatory functions where roots are regularly spaced.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = float(x0)
    
    # Helper to evaluate f safely
    def safe_f(x_val):
        try:
            fx = f(x_val)
            if not math.isfinite(fx):
                return None
            return fx
        except:
            return None

    # Check if x0 is already a root
    f0 = safe_f(x)
    if f0 is not None and abs(f0) <= 1e-8:
        return x

    # Strategy 1: Global Grid Search for Sign Change
    # We search in a range around x0. For oscillatory functions, 
    # a range of [-10, 10] relative to x0 (or absolute if x0 is small) 
    # is often sufficient. We use a step size of 0.1 to catch most oscillations.
    
    # Define search range
    # If x0 is large, search relative. If x0 is small, search absolute.
    # Let's try a range of +/- 5.0 around x0, and also +/- 5.0 absolute if x0 is near 0.
    # Actually, let's just search a fixed window around x0 first, then expand.
    
    search_ranges = [
        (x - 1.0, x + 1.0, 0.05),
        (x - 5.0, x + 5.0, 0.1),
        (x - 20.0, x + 20.0, 0.5),
        (x - 100.0, x + 100.0, 2.0),
    ]
    
    # Also try absolute ranges if x0 is close to 0
    if abs(x) < 10:
        search_ranges.append((-10.0, 10.0, 0.1))
        search_ranges.append((-50.0, 50.0, 0.5))

    best_root_candidate = x
    best_abs_f = abs(f0) if f0 is not None else float('inf')
    
    # Track function calls
    call_count = 1 # for f0
    
    for (start, end, step) in search_ranges:
        if time.time() > deadline - 5.0:
            break
            
        # Generate grid points
        n_points = int((end - start) / step) + 1
        if n_points > 1000: # Limit grid size to avoid too many calls
            step = (end - start) / 1000
            n_points = 1000
            
        prev_x = start
        prev_f = safe_f(prev_x)
        call_count += 1
        
        if prev_f is not None:
            if abs(prev_f) < best_abs_f:
                best_abs_f = abs(prev_f)
                best_root_candidate = prev_x
        
        for i in range(1, n_points):
            curr_x = start + i * step
            curr_f = safe_f(curr_x)
            call_count += 1
            
            if curr_f is None:
                prev_x, prev_f = curr_x, None
                continue
                
            if abs(curr_f) < best_abs_f:
                best_abs_f = abs(curr_f)
                best_root_candidate = curr_x
                
            if prev_f is not None:
                # Check for sign change
                if prev_f * curr_f < 0:
                    # Found a bracket [prev_x, curr_x]
                    root = brent_method(f, prev_x, curr_x, call_count, deadline)
                    if root is not None:
                        return root
                    # If brent failed, continue searching or fallback
                elif curr_f == 0:
                    return curr_x
                    
            prev_x, prev_f = curr_x, curr_f
            
            if time.time() > deadline - 5.0:
                break

    # Strategy 2: Fallback if no sign change found (e.g., tangential root or missed)
    # Use local minimization of |f(x)| around the best candidate found so far.
    # We use a simple adaptive step search (similar to Nelder-Mead but simpler).
    
    x_best = best_root_candidate
    g_best = best_abs_f
    
    # If we haven't found a good candidate, start from x0
    if g_best > 1e-8:
        # Try a local search around x0 and x_best
        candidates = [x0, best_root_candidate]
        
        for cx in candidates:
            if time.time() > deadline - 5.0:
                break
                
            # Local search with shrinking step
            step = 1.0
            x_curr = cx
            f_curr = safe_f(x_curr)
            if f_curr is None:
                continue
            g_curr = abs(f_curr)
            
            # Simple hill climbing / local search
            for _ in range(50):
                if time.time() > deadline - 2.0:
                    break
                    
                improved = False
                for direction in [1, -1]:
                    x_try = x_curr + direction * step
                    f_try = safe_f(x_try)
                    if f_try is None:
                        continue
                    g_try = abs(f_try)
                    
                    if g_try < g_curr:
                        x_curr = x_try
                        f_curr = f_try
                        g_curr = g_try
                        improved = True
                        if g_curr <= 1e-8:
                            return x_curr
                        # Shrink step if we moved
                        step *= 0.5
                        break # Try other direction next iter? Or break to re-evaluate?
                        # Let's try the other direction too if we have time
                    else:
                        # If this direction didn't improve, try the other
                        pass
                
                if not improved:
                    # No improvement in either direction with current step
                    step *= 0.1
                    if step < 1e-10:
                        break
                        
                if g_curr < g_best:
                    g_best = g_curr
                    x_best = x_curr

    # Final check
    if g_best <= 1e-8:
        return x_best
    
    # If still not good, try one last Newton-like step if we have derivative info? No.
    # Return the best we found.
    return x_best


def brent_method(f, a, b, initial_calls, deadline):
    """
    Implement Brent's method for finding a root in [a, b] where f(a)*f(b) < 0.
    Returns the root or None if it fails.
    """
    try:
        fa = f(a)
        fb = f(b)
        # fa and fb should have opposite signs
        if fa * fb > 0:
            return None
        
        # Initialize
        c = a
        fc = fa
        d = b - a
        e = d
        
        for i in range(50): # Max 50 iterations
            if time.time() > deadline - 2.0:
                return None
                
            if fb * fc > 0:
                c = a
                fc = fa
                d = b - a
                e = d
                
            if abs(fc) < abs(fb):
                a = b
                b = c
                c = a
                fa = fb
                fb = fc
                fc = fa
                
            tol = 2 * 2.220446049250313e-16 * abs(b) + 1e-14
            m = 0.5 * (c - b)
            
            if abs(m) < tol or fb == 0:
                return b
                
            if abs(e) >= tol and abs(fa) > abs(fb):
                # Inverse quadratic interpolation
                s = fb / fa
                p = 2 * m * s
                q = 1 - s
                if s > 0:
                    q = -q
                p = p - q * (fb - fc) / (fc - fa)
                q = q * (c - b)
                r = p / q
                # Check if r is within bounds and not too close to b
                if ((r < 0.5 * q and r < q) or (r >= 0.5 * q and r >= q) or 
                    (r * (c - b) < 0 and r * (c - b) >= 0) or 
                    (r * (c - b) < 0 and abs(r) >= abs(e))):
                    r = m
                else:
                    e = d
                    d = r
            else:
                # Bisection
                r = m
                e = d
                d = m
                
            if abs(d) < tol:
                return b
            else:
                d = r
                
            c = b
            fc = fb
            b = b + d
            fb = f(b)
            
            if abs(fb) <= 1e-8:
                return b
                
    except:
        return None
    
    return None

# EVOLVE-BLOCK-END
