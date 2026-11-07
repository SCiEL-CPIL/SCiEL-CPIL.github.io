import math
import time
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin before 120s limit

    x = float(x0)
    fx = f(x)
    
    # Check if x0 is already a root
    if abs(fx) <= 1e-8:
        return x

    # Strategy: Find a bracket [a, b] where f(a) and f(b) have opposite signs.
    # Then use Brent's method to converge quickly.
    # We will test points x0 + delta and x0 - delta
    # delta starts small and grows exponentially
    
    # Start with a finer grid to catch high-frequency oscillations, then expand.
    # We also track the best point (min |f|) in case we fail to bracket.
    
    best_x = x
    best_f = fx
    best_abs = abs(fx)
    
    a = x
    fa = fx
    
    # A set of deltas to try. We start very small to handle high frequency, 
    # then expand. We also include some "random-ish" or non-uniform steps 
    # to avoid missing sign changes that fall exactly between grid points 
    # if the function is very steep or has narrow roots.
    
    # Note: For oscillatory functions, the period T might be small.
    # If we step by d, we need d < T/2 to be safe, but we don't know T.
    # However, if we miss a sign change, we might still find a point with low |f|.
    # But the requirement is |f(x)| <= 1e-8.
    # If we find a bracket, Brent's method is very robust.
    
    # Let's use a logarithmic sweep.
    # Start at 1e-6 to catch very tight oscillations, up to 1e6.
    # This is 12 orders of magnitude.
    # We'll check both +d and -d.
    
    # Optimization: If we find a point with very small |f|, we can try to refine it 
    # using a local Newton step or just return it if it's good enough.
    
    bracket_found = False
    lo = 0.0
    hi = 0.0
    flo = 0.0
    fhi = 0.0
    
    # We'll iterate through a list of deltas.
    # To be efficient, we stop early if we find a bracket.
    
    deltas = []
    # Generate logarithmic steps
    for i in range(-6, 7):  # 1e-6 to 1e6
        deltas.append(10.0 ** i)
    
    # Add some intermediate steps to be safe? 
    # Log steps are 10x apart. If period is 5, and we step 1, then 10, we might miss.
    # Let's use a factor of 3 or 5 instead of 10 to be safer.
    # Or just use a list of specific values.
    
    # Let's stick to a robust list:
    # 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000, 1000000
    # This is what we had. Let's try to make it more dense near 0?
    # Maybe start at 1e-8?
    
    # Let's try a slightly different approach:
    # 1. Check immediate neighbors with very small steps.
    # 2. If no sign change, expand.
    
    # Revised delta list:
    # Very small: 1e-8, 1e-7, 1e-6
    # Small: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
    # Medium: 1, 10, 100, 1000
    # Large: 10000, 100000, 1000000
    
    # Actually, the previous list was fine. Let's just ensure we check both sides carefully.
    
    for d in deltas:
        if time.time() > deadline:
            break
            
        # Check right side
        x_right = x + d
        try:
            fr = f(x_right)
        except:
            fr = None
        
        if fr is not None and math.isfinite(fr):
            abs_fr = abs(fr)
            if abs_fr < best_abs:
                best_abs = abs_fr
                best_x = x_right
                best_f = fr
            if abs_fr <= 1e-8:
                return x_right
            if fa * fr < 0:
                lo, hi = x, x_right
                flo, fhi = fa, fr
                bracket_found = True
                break
        
        # Check left side
        x_left = x - d
        try:
            fl = f(x_left)
        except:
            fl = None
            
        if fl is not None and math.isfinite(fl):
            abs_fl = abs(fl)
            if abs_fl < best_abs:
                best_abs = abs_fl
                best_x = x_left
                best_f = fl
            if abs_fl <= 1e-8:
                return x_left
            if fa * fl < 0:
                lo, hi = x_left, x
                flo, fhi = fl, fa
                bracket_found = True
                break
        
        if time.time() > deadline:
            break

    if bracket_found:
        # Use Brent's method for fast, robust convergence
        try:
            root = brentq(f, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=100)
            if time.time() < deadline:
                # Verify
                f_root = f(root)
                if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass
        
        # If brentq failed or wasn't good enough, try bisection manually?
        # Or just return best_x?
        # Let's try a few bisection steps if we have time.
        # But we might have lost flo/fhi if brentq modified them? No, brentq takes f, lo, hi.
        # We still have lo, hi. We can re-eval or use stored.
        # Stored flo/fhi correspond to lo/hi at the start of brentq.
        # If brentq failed, the bracket might still be valid.
        
        # Let's do a simple bisection fallback
        a_b, b_b = min(lo, hi), max(lo, hi)
        fa_b, fb_b = f(a_b), f(b_b) # Re-eval to be safe
        if fa_b * fb_b < 0:
            for _ in range(20):
                if time.time() > deadline:
                    break
                m = (a_b + b_b) / 2.0
                fm = f(m)
                if abs(fm) <= 1e-8:
                    return m
                if fa_b * fm < 0:
                    b_b = m
                    fb_b = fm
                else:
                    a_b = m
                    fa_b = fm
            # Return midpoint
            m = (a_b + b_b) / 2.0
            return m

    # If no bracket found, we have the best_x found during the scan.
    # Try to refine best_x using Newton's method if possible.
    # This is useful if the function has a minimum close to 0 but no sign change nearby 
    # (e.g., touch root) or if we missed a narrow sign change.
    
    if best_abs > 1e-8:
        # Try Newton's method from best_x
        # We need an estimate of the derivative.
        # Use central difference if possible.
        
        h = 1e-5 * max(1.0, abs(best_x))
        try:
            # Check if we can evaluate neighbors
            fx_best = f(best_x)
            fx_plus = f(best_x + h)
            fx_minus = f(best_x - h)
            
            if math.isfinite(fx_plus) and math.isfinite(fx_minus):
                d = (fx_plus - fx_minus) / (2 * h)
                if d != 0 and math.isfinite(d):
                    # Newton step
                    x_new = best_x - fx_best / d
                    if math.isfinite(x_new):
                        f_new = f(x_new)
                        if math.isfinite(f_new) and abs(f_new) < best_abs:
                            best_x = x_new
                            best_f = f_new
                            best_abs = abs(f_new)
                            
                            # Try one more Newton step?
                            if best_abs > 1e-8 and time.time() < deadline:
                                fx_best2 = f(best_x)
                                fx_plus2 = f(best_x + h)
                                fx_minus2 = f(best_x - h)
                                if math.isfinite(fx_plus2) and math.isfinite(fx_minus2):
                                    d2 = (fx_plus2 - fx_minus2) / (2 * h)
                                    if d2 != 0 and math.isfinite(d2):
                                        x_new2 = best_x - fx_best2 / d2
                                        if math.isfinite(x_new2):
                                            f_new2 = f(x_new2)
                                            if math.isfinite(f_new2) and abs(f_new2) < best_abs:
                                                best_x = x_new2
                                                best_f = f_new2
                                                best_abs = abs(f_new2)
        except:
            pass

    return best_x
# EVOLVE-BLOCK-END
