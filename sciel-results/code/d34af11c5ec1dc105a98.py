"""Root finder for 3 fixed oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f (oscillatory) starting near x0.
    
    Strategy:
      1. Evaluate f at x0; if |f| <= tol, return immediately.
      2. Build a cache of (x, f(x)) evaluations.
      3. Use a dense uniform grid scan to find sign-change brackets.
      4. Apply safeguarded secant/bisection within the tightest bracket.
      5. If no bracket, use secant from best pair.
      6. Deadline guard: return best-so-far before wall-clock limit.
    """
    deadline = time.time() + 140.0  # safety margin under 150s
    tol = 1e-8
    x0 = float(x0)
    
    # Cache of evaluations: dict mapping x -> f(x)
    cache = {}
    
    # Track global best (smallest |f|)
    best_x = x0
    best_f = None
    best_abs = float('inf')
    
    def update_best(xv, fv):
        nonlocal best_x, best_f, best_abs
        if not math.isfinite(xv) or not math.isfinite(fv):
            return
        af = abs(fv)
        if af < best_abs:
            best_x = xv
            best_f = fv
            best_abs = af
    
    def eval_f(xv):
        """Evaluate f(xv) and cache the result."""
        if xv in cache:
            return cache[xv]
        if time.time() > deadline:
            return None
        try:
            fv = f(xv)
        except Exception:
            return None
        if not math.isfinite(fv):
            return None
        cache[xv] = fv
        update_best(xv, fv)
        return fv
    
    # Initial evaluation at x0
    f0 = eval_f(x0)
    if f0 is not None and abs(f0) <= tol:
        return x0
    
    # Dense uniform grid scan to find brackets
    # Use multiple scales of grid spacing
    grid_scales = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    points_per_scale = 20  # points on each side
    
    for scale in grid_scales:
        if time.time() > deadline:
            break
        
        # Generate uniform grid points around x0
        for i in range(1, points_per_scale + 1):
            for sign in [-1, 1]:
                xv = x0 + sign * i * scale
                if time.time() > deadline:
                    break
                eval_f(xv)
        
        # Check if we've found a good enough point
        if best_abs <= tol:
            return best_x
        
        # Look for sign-change brackets in the cache
        sorted_pts = sorted(cache.items(), key=lambda kv: kv[0])
        
        brackets = []
        for i in range(len(sorted_pts) - 1):
            xa, fa = sorted_pts[i]
            xb, fb = sorted_pts[i + 1]
            if fa * fb < 0:
                brackets.append((xa, xb, fa, fb))
        
        if brackets:
            # Sort brackets by span (tighter first)
            brackets.sort(key=lambda t: abs(t[1] - t[0]))
            
            for (a, b, fa, fb) in brackets:
                if time.time() > deadline:
                    break
                
                # Safeguarded secant method within the bracket
                xa, xb = a, b
                fa, fb = fa, fb
                
                for _ in range(80):
                    if time.time() > deadline:
                        break
                    
                    if abs(fa) <= tol:
                        return xa
                    if abs(fb) <= tol:
                        return xb
                    
                    # Try secant step
                    denom = fb - fa
                    if abs(denom) > 1e-15:
                        xn = xb - fb * (xb - xa) / denom
                    else:
                        xn = 0.5 * (xa + xb)
                    
                    # Safeguard: if secant step goes outside bracket, use bisection
                    if not (min(xa, xb) <= xn <= max(xa, xb)):
                        xn = 0.5 * (xa + xb)
                    
                    if xn == xa or xn == xb:
                        # Interval too small
                        if abs(fa) <= abs(fb):
                            return xa
                        else:
                            return xb
                    
                    fn = eval_f(xn)
                    if fn is None:
                        break
                    
                    if abs(fn) <= tol:
                        return xn
                    
                    # Update bracket
                    if fa * fn <= 0:
                        xb, fb = xn, fn
                    else:
                        xa, fa = xn, fn
                
                # After convergence attempt, return best endpoint
                if abs(fa) <= abs(fb):
                    return xa
                else:
                    return xb
    
    # No bracket found; try secant from the two best points
    if len(cache) >= 2:
        sorted_by_f = sorted(cache.items(), key=lambda kv: abs(kv[1]))
        (x1, f1), (x2, f2) = sorted_by_f[0], sorted_by_f[1]
        
        if x1 != x2 and f1 != f2:
            xa, xb = x1, x2
            fa, fb = f1, f2
            
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                if abs(fa) <= tol:
                    return xa
                if abs(fb) <= tol:
                    return xb
                
                denom = fb - fa
                if abs(denom) < 1e-15:
                    break
                
                xn = xb - fb * (xb - xa) / denom
                if not math.isfinite(xn):
                    break
                
                fn = eval_f(xn)
                if fn is None:
                    break
                
                if abs(fn) <= tol:
                    return xn
                
                xa, fa = xb, fb
                xb, fb = xn, fn
    
    return best_x
# EVOLVE-BLOCK-END
