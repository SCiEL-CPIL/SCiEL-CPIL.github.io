"""Single-shot cubic reconstruction via Lagrange interpolation + numpy.roots."""
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f near x0 using cubic polynomial reconstruction.
    Strategy:
      1. Evaluate f at 4 well-chosen points around x0.
      2. Fit a cubic polynomial through these points.
      3. Find all real roots of the cubic.
      4. Pick the real root nearest to x0 that satisfies |f(x)| <= 1e-8.
      5. If no good root found, adaptively widen the search window and repeat.
      6. Final fallback: Newton's method.
    """
    x = float(x0)
    
    # Adaptive window sizes to try, starting small
    window_scales = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    
    for scale in window_scales:
        # Choose 4 points: two on each side of x, with asymmetric spacing for better conditioning
        # Using Chebyshev-like nodes scaled to the window
        h = scale * max(1.0, abs(x))
        
        # Four points: roughly at -1.5h, -0.5h, +0.5h, +1.5h relative to x
        # But use slightly different offsets to avoid symmetry issues
        offsets = [-1.4 * h, -0.4 * h, 0.4 * h, 1.4 * h]
        pts = [x + off for off in offsets]
        
        # Ensure all points are distinct
        if len(set(pts)) < 4:
            # Perturb slightly
            pts = [x - 1.5*h, x - 0.5*h, x + 0.5*h, x + 1.5*h]
        
        # Evaluate f at the 4 points
        try:
            vals = [f(p) for p in pts]
        except Exception:
            continue
        
        # Check for non-finite values
        if any(not np.isfinite(v) for v in vals):
            continue
        
        # Fit cubic polynomial using numpy.polyfit
        try:
            coeffs = np.polyfit(np.array(pts), np.array(vals), 3)
        except Exception:
            continue
        
        # Find roots of the cubic
        try:
            roots = np.roots(coeffs)
        except Exception:
            continue
        
        # Filter for real roots
        real_roots = []
        for r in roots:
            if np.isfinite(r.real) and abs(r.imag) < 1e-9:
                real_roots.append(float(r.real))
        
        if not real_roots:
            continue
        
        # Pick the real root closest to x
        best = min(real_roots, key=lambda r: abs(r - x))
        
        # Verify the root
        try:
            fb = f(best)
        except Exception:
            continue
        
        if np.isfinite(fb) and abs(fb) <= 1e-8:
            return float(best)
    
    # If all interpolation attempts failed, use Newton's method as final fallback
    x = float(x0)
    for _ in range(80):
        try:
            fx = f(x)
        except Exception:
            break
        
        if not np.isfinite(fx):
            break
        
        if abs(fx) <= 1e-12:
            return float(x)
        
        # Numerical derivative
        h = 1e-7 * max(1.0, abs(x))
        try:
            fxp = f(x + h)
        except Exception:
            break
        
        if not np.isfinite(fxp):
            break
        
        d = (fxp - fx) / h
        if abs(d) < 1e-15:
            break
        
        x_new = x - fx / d
        if not np.isfinite(x_new):
            break
        
        if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
            # Converged
            try:
                fx_final = f(x_new)
                if np.isfinite(fx_final) and abs(fx_final) <= 1e-8:
                    return float(x_new)
            except Exception:
                pass
            break
        
        x = x_new
    
    return float(x)
# EVOLVE-BLOCK-END
