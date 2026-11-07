"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq, newton

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    x0 = float(x0)
    call_count = 0
    max_calls = 480  # Stay well under 500
    
    best_x = x0
    best_fx = None
    
    def f_wrapper(x):
        nonlocal call_count, best_x, best_fx
        if call_count >= max_calls:
            raise Exception("Call limit exceeded")
        if time.time() > deadline:
            raise Exception("Time limit exceeded")
            
        call_count += 1
        try:
            fx = f(x)
        except Exception:
            return float('nan')
        
        if math.isfinite(fx):
            if best_fx is None or abs(fx) < abs(best_fx):
                best_fx = fx
                best_x = x
                
        return fx

    # Initial evaluation
    try:
        fx0 = f_wrapper(x0)
    except Exception:
        return best_x

    if not math.isfinite(fx0):
        return best_x
        
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy 1: Try to find a bracket and use Brent's method
    bracket_found = False
    a = x0
    b = x0
    fa = fx0
    fb = None
    
    # Try expanding in both directions
    delta = 1e-3
    for _ in range(40):
        if call_count >= max_calls - 50:
            break
        if time.time() > deadline:
            break
            
        # Check x0 + delta
        try:
            b_candidate = x0 + delta
            fb_candidate = f_wrapper(b_candidate)
        except Exception:
            break
            
        if math.isfinite(fb_candidate):
            if abs(fb_candidate) <= 1e-8:
                return b_candidate
            if fa * fb_candidate < 0:
                bracket_found = True
                a = x0
                b = b_candidate
                break
        else:
            # If non-finite, maybe overflow? Try smaller delta? 
            # Or just skip this direction?
            pass
            
        # Check x0 - delta
        try:
            a_candidate = x0 - delta
            fa_candidate = f_wrapper(a_candidate)
        except Exception:
            break
            
        if math.isfinite(fa_candidate):
            if abs(fa_candidate) <= 1e-8:
                return a_candidate
            if fa * fa_candidate < 0:
                bracket_found = True
                a = a_candidate
                b = x0
                break
        
        delta *= 2.0
        if delta > 1e8:
            break

    if bracket_found:
        try:
            # Ensure a < b
            if a > b:
                a, b = b, a
                
            root = brentq(f_wrapper, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                # Verify
                fr = f_wrapper(root)
                if math.isfinite(fr) and abs(fr) <= 1e-8:
                    return root
                # If brentq converged but |f(root)| > 1e-8 (unlikely), 
                # return root if it's better than best_x?
                # Or just return best_x?
                # brentq guarantees f(root) is close to 0.
                return root
        except Exception:
            pass

    # Strategy 2: Use Newton/Secant method
    try:
        # newton with fprime=None uses secant
        root = newton(f_wrapper, x0, tol=1e-12, maxiter=100)
        if math.isfinite(root):
            fr = f_wrapper(root)
            if math.isfinite(fr) and abs(fr) <= 1e-8:
                return root
            # If secant found a good point but not < 1e-8, return it?
            # The task requires |f(x)| <= 1e-8.
            # If newton didn't reach it, best_x might be better.
            # But newton usually converges.
            if abs(fr) < abs(best_fx) if best_fx is not None else True:
                return root
    except Exception:
        pass

    # Fallback: return best found
    return best_x
# EVOLVE-BLOCK-END
