import numpy as np
import scipy.optimize as opt
import time
import math

def find_root(f, x0):
    """
    Finds a root of f starting near x0.
    Strategy:
    1. Try single-shot cubic reconstruction (Lagrange/Polyfit) with 4 points.
    2. If fails, use Brent's method with bracket expansion.
    3. If fails, use Newton/Secant hybrid.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well under 120s limit
    
    x0 = float(x0)
    
    # --- Phase 1: Single-shot Cubic Reconstruction ---
    # We assume the polynomial might be cubic or lower.
    # We evaluate at 4 distinct points near x0.
    # Step size h should be small enough for local approximation to hold if degree <= 3,
    # but large enough to be numerically stable.
    # If the function is exactly cubic, this will find the root analytically.
    
    # Choose points: x0, x0+h, x0+2h, x0+3h
    # We need to pick h carefully.
    # If x0 is large, h should be large.
    # Let's try a relative step.
    h = 1e-3 * max(1.0, abs(x0))
    
    try:
        # Check for immediate root
        f0 = f(x0)
        if abs(f0) <= 1e-8:
            return x0
            
        # Define 4 points
        pts = [x0, x0 + h, x0 + 2*h, x0 + 3*h]
        vals = []
        for p in pts:
            try:
                vp = f(p)
                if not math.isfinite(vp):
                    raise ValueError("Non-finite value")
                vals.append(vp)
            except Exception:
                # If f fails, fallback
                raise Exception("f failed in Phase 1")

        # Fit cubic polynomial
        # coeffs = np.polyfit(pts, vals, 3)
        # roots = np.roots(coeffs)
        # This is stable enough for 4 points.
        coeffs = np.polyfit(pts, vals, 3)
        roots = np.roots(coeffs)
        
        # Filter real roots
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        
        if real_roots:
            # Find root closest to x0
            best_x = min(real_roots, key=lambda r: abs(r - x0))
            
            # Verify
            try:
                f_best = f(best_x)
                if math.isfinite(f_best) and abs(f_best) <= 1e-8:
                    return best_x
            except Exception:
                pass
    
    except Exception:
        pass

    # --- Phase 2: Bracketing + Brent's Method ---
    # Find a bracket [a, b] such that f(a)*f(b) < 0
    # Start with small interval around x0
    a = x0 - 1e-3
    b = x0 + 1e-3
    fa = None
    fb = None
    
    try:
        fa = f(a)
        fb = f(b)
    except Exception:
        pass
        
    # Expand bracket if no sign change
    if fa is not None and fb is not None:
        if not math.isfinite(fa) or not math.isfinite(fb):
            # Reset if non-finite
            fa = None
            fb = None
            
    max_expansions = 50
    expansion_factor = 2.0
    width = 1e-3 * max(1.0, abs(x0))
    found_bracket = False
    
    if fa is not None and fb is not None and math.isfinite(fa) and math.isfinite(fb):
        if fa * fb < 0:
            found_bracket = True
        else:
            # Expand
            for i in range(max_expansions):
                if time.time() > deadline:
                    break
                width *= expansion_factor
                a = x0 - width
                b = x0 + width
                try:
                    fa = f(a)
                    fb = f(b)
                except Exception:
                    break
                
                if not math.isfinite(fa) or not math.isfinite(fb):
                    break
                    
                if fa * fb < 0:
                    found_bracket = True
                    break
                if abs(fa) < 1e-12:
                    return a
                if abs(fb) < 1e-12:
                    return b

    if found_bracket:
        try:
            # Use Brent's method
            root, res = opt.brentq(f, a, b, full_output=True, maxiter=50)
            if res.converged:
                f_root = f(root)
                if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass

    # --- Phase 3: Newton/Secant Hybrid Fallback ---
    # If bracketing failed (e.g., double root or no sign change in domain),
    # use Newton with fallback to Secant.
    
    # Re-use Phase 1 values if possible? No, they might be stale or inconsistent with new path.
    # Let's restart with a robust iterative method.
    
    # Start from x0
    x_curr = x0
    try:
        f_curr = f(x_curr)
    except Exception:
        return x_curr # Last resort

    if abs(f_curr) <= 1e-8:
        return x_curr

    # Try to get a second point for Secant
    h_sec = 1e-4 * max(1.0, abs(x_curr))
    x_prev = x_curr
    x_curr = x_curr + h_sec
    try:
        f_prev = f(x_prev)
        f_curr = f(x_curr)
    except Exception:
        return x_prev if abs(f_prev) <= 1e-8 else x_prev # Return best guess

    # Hybrid Newton-Secant
    # We don't have derivative, so we rely on Secant or approximate Newton.
    # Secant is O(1.618), Newton is O(2).
    # Since we failed to bracket, the function might not cross zero,
    # but we need |f(x)| <= 1e-8.
    
    # Use a simple iterative improvement
    for _ in range(50):
        if time.time() > deadline:
            break
        if abs(f_curr) <= 1e-8:
            return x_curr
            
        if abs(f_prev) < abs(f_curr):
            # Swap to keep best guess
            x_prev, x_curr = x_curr, x_prev
            f_prev, f_curr = f_curr, f_prev
            
        if f_curr == f_prev:
            break
            
        # Secant step
        slope = (f_curr - f_prev) / (x_curr - x_prev)
        if slope == 0:
            break
        x_new = x_curr - f_curr / slope
        
        # Check for non-finite
        if not math.isfinite(x_new):
            break
            
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        
        try:
            f_curr = f(x_curr)
        except Exception:
            break
        
    return x_curr
