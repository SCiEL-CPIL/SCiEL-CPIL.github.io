"""
Strategy: Adaptive Secant Exploration with Fallback Bisection
1. Primary Phase: Use the Secant Method starting from x0 and x0 + delta.
   The Secant method has superlinear convergence (order ~1.618) for smooth
   functions, often finding roots in 5-15 calls. This is much faster than
   grid-based bracketing.
2. Safety Net: If the Secant method diverges, oscillates, or fails to
   converge within a small budget (e.g., 50 calls), we switch to a
   robust local bisection search.
3. Local Bisection: Perform a fine-grained linear sweep around the last
   known good region (or x0) to find a sign change, then refine with
   Illinois/Bisection.
4. Budget Management: Strictly limit total calls to ~300 to maximize the
   score component (1 - 0.5 * log10(mean_calls) / log10(500)).
   Lower mean calls significantly boost the score if success rate is maintained.
"""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    MAX_CALLS = 300     # Aggressive budget to boost score
    TOL = 1e-9          # Target tolerance
    
    calls = 0
    cache = {}
    
    def _f(x):
        nonlocal calls
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        key = x
        if key in cache:
            return cache[key]
            
        if calls >= MAX_CALLS:
            raise RuntimeError("Call limit exceeded")
            
        calls += 1
        val = f(x)
        cache[key] = val
        return val

    # State for best result found so far
    best_x = x0
    best_f_val = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_f_val
        if math.isfinite(x) and math.isfinite(fx):
            if abs(fx) < abs(best_f_val):
                best_x = x
                best_f_val = fx

    # 1. Initial check
    try:
        fx0 = _f(x0)
        if abs(fx0) <= TOL:
            return x0
        update_best(x0, fx0)
    except Exception:
        return x0

    # 2. Primary Phase: Secant Method
    # Start with x0 and a perturbed point
    # Choose delta based on x0 scale
    scale = max(1.0, abs(x0))
    delta = scale * 1e-3
    
    x_prev = x0
    x_curr = x0 + delta
    
    try:
        f_prev = fx0
        f_curr = _f(x_curr)
        update_best(x_curr, f_curr)
        
        # Secant iteration
        # Limit secant iterations to 30 to prevent runaway
        for i in range(30):
            if time.time() - start_time > TIME_LIMIT:
                break
            if calls >= MAX_CALLS - 50: # Leave budget for fallback
                break
                
            if abs(f_curr) <= TOL:
                return x_curr
            if abs(f_prev) <= TOL:
                return x_prev
                
            # Check for division by zero
            if abs(f_curr - f_prev) < 1e-15:
                break
                
            # Secant formula
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            # Sanity check: ensure x_new is finite
            if not math.isfinite(x_new):
                break
                
            # Optional: Damping if step is too large
            # If |x_new - x_curr| > 10 * scale, dampen
            step = abs(x_new - x_curr)
            if step > 10 * scale:
                # Dampen
                x_new = x_curr + math.copysign(10 * scale, x_new - x_curr)
                
            f_new = _f(x_new)
            update_best(x_new, f_new)
            
            if abs(f_new) <= TOL:
                return x_new
                
            # Shift
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new
            
            # Check if we've converged sufficiently
            if abs(x_curr - x_prev) < 1e-12 * scale:
                 if abs(f_curr) < abs(f_prev):
                     return x_curr
                 else:
                     return x_prev
                     
    except Exception:
        pass

    # 3. Fallback Phase: Local Dense Sweep
    # If secant failed, we need to find a bracket.
    # Search in a window around x0 and the last secant points.
    # Window size: max(1.0, abs(x0)*0.1)
    win = max(1.0, abs(x0) * 0.1)
    
    # Define search range
    # We'll try a few different windows if the first one fails
    # But to save calls, let's try a dense grid in the primary window.
    
    # We have used ~30-50 calls in secant. We have ~250 left.
    # Let's use 100 calls for the sweep.
    N_SWEEP = 100
    a = x0 - win
    b = x0 + win
    
    # We might already have points in this range from secant.
    # But for simplicity and robustness, let's do a fresh sweep 
    # using the cache to avoid double-counting if points match exactly.
    
    step = (b - a) / N_SWEEP
    prev_x = a
    prev_fx = None
    
    try:
        for i in range(N_SWEEP + 1):
            cur_x = a + i * step
            cur_fx = _f(cur_x)
            update_best(cur_x, cur_fx)
            
            if prev_fx is not None:
                if prev_fx * cur_fx < 0:
                    # Found bracket
                    bracket_a = prev_x
                    bracket_b = cur_x
                    fa = prev_fx
                    fb = cur_fx
                    
                    # 4. Refinement: Illinois Method
                    fa_w = fa
                    fb_w = fb
                    a_b = bracket_a
                    b_b = bracket_b
                    
                    for _ in range(100):
                        if time.time() - start_time > TIME_LIMIT:
                            break
                        if calls >= MAX_CALLS:
                            break
                            
                        denom = abs(fa_w) + abs(fb_w)
                        if denom == 0:
                            break
                            
                        m = (a_b * abs(fb_w) + b_b * abs(fa_w)) / denom
                        
                        try:
                            fm = _f(m)
                        except Exception:
                            break
                            
                        update_best(m, fm)
                        
                        if abs(fm) <= TOL:
                            return m
                            
                        if fa * fm < 0:
                            b_b = m
                            fb = fm
                            fb_w = fm / 2.0
                        else:
                            a_b = m
                            fa = fm
                            fa_w = fm / 2.0
                            
                        if abs(b_b - a_b) < 1e-12:
                            break
                            
                    # Return best of the last few
                    candidates = [(a_b, fa), (b_b, fb), (m, fm)]
                    best_cand = min(candidates, key=lambda x: abs(x[1]))
                    return best_cand[0]
            
            prev_x = cur_x
            prev_fx = cur_fx
            
    except Exception:
        pass

    # If no bracket found, return best_x
    if math.isfinite(best_x):
        return best_x
    return x0

# EVOLVE-BLOCK-END
