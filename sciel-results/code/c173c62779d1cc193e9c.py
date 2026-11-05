"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import brentq
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    MODEL-AND-OPTIMIZE approach:
    - Fit local quadratic models to recent samples to predict root locations.
    - Track a global bracket and use Brent's method when a sign change is found.
    - Multi-scale exploration with deadline guard.
    """
    x0 = float(x0)
    deadline = time.time() + 115  # 5s safety margin before 120s kill
    
    # State
    eval_count = 0
    max_evals = 500
    
    def f_counted(x):
        nonlocal eval_count
        if time.time() > deadline:
            raise TimeoutError
        eval_count += 1
        return f(x)
    
    best_x = x0
    fx0 = f_counted(x0)
    best_f = abs(fx0)
    if best_f <= 1e-8:
        return best_x
    
    # Illinois algorithm: robust bracketing method for ill-conditioned problems
    def illinois_solve(a, b, fa, fb):
        nonlocal best_x, best_f
        if fa == 0:
            return a
        if fb == 0:
            return b
        if fa * fb > 0:
            return None
        try:
            lo, hi = a, b
            flo, fhi = fa, fb
            for _ in range(50):
                if time.time() > deadline:
                    break
                if flo == 0:
                    return lo
                if fhi == 0:
                    return hi
                if flo * fhi > 0:
                    break
                # Regula falsi step
                if abs(flo - fhi) > 1e-300:
                    xs = hi - fhi * (hi - lo) / (fhi - flo)
                else:
                    xs = (lo + hi) / 2.0
                # Safeguard: if outside bracket, use midpoint
                if not (lo < xs < hi):
                    xs = (lo + hi) / 2.0
                fx = f_counted(xs)
                if abs(fx) < best_f:
                    best_x = xs
                    best_f = abs(fx)
                if fx == 0:
                    return xs
                if flo * fx < 0:
                    hi = xs
                    fhi = fx
                else:
                    lo = xs
                    flo = fx
                # Illinois: shrink the endpoint with larger |f| by half
                if abs(flo) > abs(fhi):
                    lo = (lo + hi) / 2.0
                    flo = f_counted(lo)
                    if abs(flo) < best_f:
                        best_x = lo
                        best_f = abs(flo)
                else:
                    hi = (lo + hi) / 2.0
                    fhi = f_counted(hi)
                    if abs(fhi) < best_f:
                        best_x = hi
                        best_f = abs(fhi)
                if abs(hi - lo) < 1e-12 * max(1.0, abs(lo), abs(hi)):
                    break
            # Return the better endpoint
            if abs(flo) < abs(fhi):
                return lo
            else:
                return hi
        except Exception:
            return None
    
    # Model-and-optimize: fit quadratic model to predict root
    def model_predict(points):
        """
        Fit a quadratic model y = a*x^2 + b*x + c to points and
        predict where it crosses zero near the bracket or best point.
        points: list of (x, f(x)) pairs
        Returns candidate x or None
        """
        if len(points) < 2:
            return None
        if len(points) < 3:
            # Linear model
            (x1, y1), (x2, y2) = points[-2:]
            if abs(y1 - y2) < 1e-300:
                return None
            # Root of linear: x = x1 - y1*(x2-x1)/(y2-y1)
            xr = x1 - y1 * (x2 - x1) / (y2 - y1)
            return xr
        # Quadratic model from last 3 points
        (x1, y1), (x2, y2), (x3, y3) = points[-3:]
        # Check for degenerate cases
        if abs(x1 - x2) < 1e-15 or abs(x2 - x3) < 1e-15 or abs(x1 - x3) < 1e-15:
            return None
        # Lagrange interpolation coefficients for quadratic
        # y = a*x^2 + b*x + c
        # Solve 3x3 system
        A = np.array([[x1**2, x1, 1],
                      [x2**2, x2, 1],
                      [x3**2, x3, 1]], dtype=float)
        b_vec = np.array([y1, y2, y3], dtype=float)
        try:
            coeffs = np.linalg.solve(A, b_vec)
            a, b, c = coeffs
        except np.linalg.LinAlgError:
            return None
        # Find roots of a*x^2 + b*x + c = 0
        if abs(a) < 1e-300:
            # Linear
            if abs(b) < 1e-300:
                return None
            xr = -c / b
        else:
            disc = b*b - 4*a*c
            if disc < -1e-12:
                return None
            disc = max(0, disc)
            sqrt_disc = np.sqrt(disc)
            r1 = (-b + sqrt_disc) / (2*a)
            r2 = (-b - sqrt_disc) / (2*a)
            # Choose the root closer to best_x or within a reasonable range
            # Prefer root closer to the cluster of points
            xmid = (x1 + x2 + x3) / 3.0
            if abs(r1 - xmid) < abs(r2 - xmid):
                xr = r1
            else:
                xr = r2
        # Sanity check: don't go too far
        xref = best_x
        if abs(xr - xref) > 100 * max(1.0, abs(xref)):
            return None
        return xr
    
    # Exploration strategy
    def explore(x_start, scale):
        nonlocal best_x, best_f, eval_count
        if time.time() > deadline or eval_count >= max_evals - 5:
            return False
        a = x_start - scale
        b = x_start + scale
        fa = f_counted(a)
        fb = f_counted(b)
        if abs(fa) < best_f:
            best_x = a
            best_f = abs(fa)
        if abs(fb) < best_f:
            best_x = b
            best_f = abs(fb)
        if best_f <= 1e-8:
            return True
        if fa * fb < 0:
            root = try_brent(a, b, fa, fb)
            if root is not None:
                fr = f_counted(root)
                if abs(fr) <= 1e-8:
                    best_x = root
                    best_f = abs(fr)
                    return True
                if abs(fr) < best_f:
                    best_x = root
                    best_f = abs(fr)
            return True  # Found bracket and tried it
        return False
    
    # Collect points for model fitting - track actual signed f values
    points = [(x0, fx0)]
    
    # Phase 1: Local expansion from x0 to find a bracket
    scales = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    for s in scales:
        if time.time() > deadline or eval_count >= max_evals - 10:
            break
        a = x0 - s
        b = x0 + s
        if eval_count >= max_evals - 5:
            break
        fa = f_counted(a)
        fb = f_counted(b)
        points.append((a, fa))
        points.append((b, fb))
        if abs(fa) < best_f:
            best_x = a
            best_f = abs(fa)
        if abs(fb) < best_f:
            best_x = b
            best_f = abs(fb)
        if best_f <= 1e-8:
            return best_x
        # Check for sign change with x0 or between a and b
        if fa * fx0 < 0:
            root = illinois_solve(a, x0, fa, fx0)
            if root is not None:
                fr = f_counted(root)
                if abs(fr) <= 1e-8:
                    return root
                if abs(fr) < best_f:
                    best_x = root
                    best_f = abs(fr)
            break
        if fb * fx0 < 0:
            root = illinois_solve(x0, b, fx0, fb)
            if root is not None:
                fr = f_counted(root)
                if abs(fr) <= 1e-8:
                    return root
                if abs(fr) < best_f:
                    best_x = root
                    best_f = abs(fr)
            break
        if fa * fb < 0:
            root = illinois_solve(a, b, fa, fb)
            if root is not None:
                fr = f_counted(root)
                if abs(fr) <= 1e-8:
                    return root
                if abs(fr) < best_f:
                    best_x = root
                    best_f = abs(fr)
            break
    
    # Phase 2: Model-guided local search
    for iteration in range(30):
        if time.time() > deadline or eval_count >= max_evals - 10:
            break
        
        # Get model prediction
        x_pred = model_predict(points)
        
        if x_pred is None:
            # Fall back to local perturbation around best point
            xbase = best_x
            x_pred = xbase + 0.1 * max(1.0, abs(xbase)) * np.random.uniform(-1, 1)
        
        # Evaluate at predicted point
        if eval_count >= max_evals - 5:
            break
        fx = f_counted(x_pred)
        if abs(fx) < best_f:
            best_x = x_pred
            best_f = abs(fx)
        if best_f <= 1e-8:
            return best_x
        points.append((x_pred, fx))
        
        # Check if we can form a bracket with any pair of recent points
        for i in range(len(points) - 1, 0, -1):
            xi, fi = points[i]
            xj, fj = points[i-1]
            if fi * fj < 0:
                root = illinois_solve(xj, xi, fj, fi)
                if root is not None:
                    fr = f_counted(root)
                    if abs(fr) <= 1e-8:
                        return root
                    if abs(fr) < best_f:
                        best_x = root
                        best_f = abs(fr)
                break
        
        # Keep points list manageable
        if len(points) > 20:
            points = points[-10:]
    
    # Phase 3: Broader multi-scale exploration
    np.random.seed(42)
    for radius in [0.5, 1, 2, 5, 10, 50, 100, 500]:
        if time.time() > deadline or eval_count >= max_evals - 20:
            break
        for _ in range(3):
            if time.time() > deadline or eval_count >= max_evals - 10:
                break
            x_try = x0 + radius * np.random.uniform(-1, 1)
            a = x_try - radius * 0.1
            b = x_try + radius * 0.1
            if eval_count >= max_evals - 5:
                break
            fa = f_counted(a)
            fb = f_counted(b)
            points.append((a, fa))
            points.append((b, fb))
            if abs(fa) < best_f:
                best_x = a
                best_f = abs(fa)
            if abs(fb) < best_f:
                best_x = b
                best_f = abs(fb)
            if best_f <= 1e-8:
                return best_x
            # Check brackets
            if fa * fb < 0:
                root = illinois_solve(a, b, fa, fb)
                if root is not None:
                    fr = f_counted(root)
                    if abs(fr) <= 1e-8:
                        return root
                    if abs(fr) < best_f:
                        best_x = root
                        best_f = abs(fr)
                break
            # Also check against best_x
            if fa * f_counted(best_x) < 0:
                root = illinois_solve(a, best_x, fa, f_counted(best_x))
                if root is not None:
                    fr = f_counted(root)
                    if abs(fr) <= 1e-8:
                        return root
                    if abs(fr) < best_f:
                        best_x = root
                        best_f = abs(fr)
                break
    
    # Phase 4: Final model-guided refinement
    for iteration in range(20):
        if time.time() > deadline or eval_count >= max_evals - 5:
            break
        x_pred = model_predict(points)
        if x_pred is None:
            break
        if abs(x_pred - points[-1][0]) < 1e-15:
            break
        fx = f_counted(x_pred)
        if abs(fx) < best_f:
            best_x = x_pred
            best_f = abs(fx)
        if best_f <= 1e-8:
            return best_x
        points.append((x_pred, fx))
        if len(points) > 20:
            points = points[-10:]
    
    # Final check: try Illinois on any sign-change pair in points
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if points[i][1] * points[j][1] < 0:
                a, b = min(points[i][0], points[j][0]), max(points[i][0], points[j][0])
                fa, fb = points[i][1] if points[i][0] == a else points[j][1]
                fb = points[j][1] if points[j][0] == b else fa
                root = illinois_solve(a, b, fa, fb)
                if root is not None:
                    fr = f_counted(root)
                    if abs(fr) <= 1e-8:
                        return root
                    if abs(fr) < best_f:
                        best_x = root
                        best_f = abs(fr)
    
    return best_x
# EVOLVE-BLOCK-END
