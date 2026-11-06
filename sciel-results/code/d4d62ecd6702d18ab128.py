"""Direct approach for ill-conditioned root finding with multiple strategies."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    max_evals = 500
    TOL = 1e-8
    POLISH_TOL = 1e-12
    eval_count = 0
    best_x = float(x0)
    best_fabs = abs(f(best_x))
    eval_count += 1

    def call_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            raise Exception("max evals")
        if time.time() > deadline:
            raise Exception("deadline")
        val = f(x)
        eval_count += 1
        if not (isinstance(val, (int, float)) and math.isfinite(val)):
            val = float('nan')
        fabs = abs(val)
        if math.isfinite(fabs) and fabs < best_fabs:
            best_fabs = fabs
            best_x = x
        return val

    def check_time():
        return time.time() < deadline

    def secant_method(x0, tol=TOL, max_iter=200):
        """Secant method with careful handling of ill-conditioning."""
        x1 = float(x0)
        # Choose a small step to estimate second point
        h = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + h
        
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False
        
        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True
        
        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True
            
            denom = f2 - f1
            if abs(denom) < 1e-30:
                # Try a different perturbation
                h2 = 1e-4 * max(1.0, abs(x2)) * (1 + 0.1 * i)
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False
            
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
        
        return abs(f1) <= tol or abs(f2) <= tol

    def newton_forward(x0, tol=TOL, max_iter=100):
        """Newton's method with forward finite difference for derivative."""
        x = float(x0)
        
        for i in range(max_iter):
            if not check_time():
                return False
            try:
                fx = call_f(x)
            except Exception:
                return False
            if abs(fx) <= tol:
                return True
            
            # Forward difference with adaptive step
            h = 1e-6 * max(1.0, abs(x))
            try:
                fp = call_f(x + h)
            except Exception:
                return False
            
            d = (fp - fx) / h
            if abs(d) < 1e-30:
                # Derivative too small, try larger step
                h2 = 1e-3 * max(1.0, abs(x))
                try:
                    fp2 = call_f(x + h2)
                    d = (fp2 - fx) / h2
                except Exception:
                    return False
                if abs(d) < 1e-30:
                    return False
            
            step = fx / d
            # Damped step to avoid overshooting
            max_step = 5.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            x_new = x - step
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            x = x_new
        
        try:
            fx = call_f(x)
        except Exception:
            return False
        return abs(fx) <= tol

    def bisection(x0, tol=TOL, max_iter=200):
        """Bisection with bracket search."""
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= tol:
            return True
        
        # Search for a bracket with multiple perturbation scales
        base_step = 1e-3 * max(1.0, abs(x))
        a, b = None, None
        fa, fb = None, None
        
        # Try multiple scales for bracket detection
        for scale_factor in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
            step = base_step * scale_factor
            if step < 1e-15:
                step = 1e-15
            
            # Search right
            for i in range(50):
                if not check_time():
                    return False
                x_test = x + step * (2 ** i)
                try:
                    f_test = call_f(x_test)
                except Exception:
                    break
                if not math.isfinite(f_test):
                    break
                if fa is None:
                    a, fa = x, fx
                if fa * f_test < 0:
                    b, fb = x_test, f_test
                    break
            if b is not None:
                break
            
            # Search left
            if b is None:
                a, b = None, None
                fa, fb = None, None
                for i in range(50):
                    if not check_time():
                        return False
                    x_test = x - step * (2 ** i)
                    try:
                        f_test = call_f(x_test)
                    except Exception:
                        break
                    if not math.isfinite(f_test):
                        break
                    if fb is None:
                        b, fb = x, fx
                    if fb * f_test < 0:
                        a, fa = x_test, f_test
                        break
            if b is not None:
                break
        
        if a is None or b is None:
            return False
        
        # Bisection
        for i in range(max_iter):
            if not check_time():
                return False
            mid = (a + b) / 2
            try:
                fm = call_f(mid)
            except Exception:
                return False
            if abs(fm) <= tol:
                return True
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
        
        return min(abs(fa), abs(fb)) <= tol

    def nelder_mead(x0, tol=TOL, max_iter=300):
        """
        Nelder-Mead simplex method to minimize |f(x)|.
        Treats root-finding as zero-dimensional optimization.
        """
        x = float(x0)
        
        # Initialize simplex with 2 points (1D problem)
        h = 1e-4 * max(1.0, abs(x))
        x1 = x
        x2 = x + h
        
        try:
            f1 = abs(call_f(x1))
            f2 = abs(call_f(x2))
        except Exception:
            return False
        
        # Ensure x1 has lower value
        if f2 < f1:
            x1, x2 = x2, x1
            f1, f2 = f2, f1
        
        if f1 <= tol:
            return True
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        for i in range(max_iter):
            if not check_time():
                return False
            
            # Order points: worst is x2, best is x1
            if f2 < f1:
                x1, x2 = x2, x1
                f1, f2 = f2, f1
            
            # Centroid of best point (in 1D, it's just the best point)
            centroid = x1
            
            # Reflection
            x_r = centroid + alpha * (centroid - x2)
            try:
                f_r = abs(call_f(x_r))
            except Exception:
                return False
            
            if f_r < f1:
                # Try expansion
                x_e = centroid + gamma * (centroid - x2)
                try:
                    f_e = abs(call_f(x_e))
                except Exception:
                    return False
                
                if f_e < f_r:
                    x2, f2 = x_e, f_e
                else:
                    x2, f2 = x_r, f_r
            elif f_r < f2:
                x2, f2 = x_r, f_r
            else:
                # Contraction
                x_c = centroid + rho * (x2 - centroid)
                try:
                    f_c = abs(call_f(x_c))
                except Exception:
                    return False
                
                if f_c < f2:
                    x2, f2 = x_c, f_c
                else:
                    # Shrink
                    x2 = x1 + sigma * (x2 - x1)
                    try:
                        f2 = abs(call_f(x2))
                    except Exception:
                        return False
            
            # Check convergence
            if f1 <= tol:
                return True
            if f2 <= tol:
                return True
            
            # Check if simplex is too small
            if abs(x2 - x1) < 1e-15 * max(1.0, abs(x1)):
                break
        
        return min(f1, f2) <= tol

    def polish(x0, tol=TOL):
        """High-precision polish pass using secant with tighter tolerance."""
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return False
        
        if abs(fx) <= tol:
            return True
        
        # Use secant method with tight tolerance for polishing
        h = 1e-6 * max(1.0, abs(x))
        x2 = x + h
        try:
            f2 = call_f(x2)
        except Exception:
            return False
        
        x1, f1 = x, fx
        
        for i in range(50):
            if not check_time():
                return False
            
            if abs(f1) <= tol:
                return True
            if abs(f2) <= tol:
                return True
            
            denom = f2 - f1
            if abs(denom) < 1e-30:
                return False
            
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
        
        return min(abs(f1), abs(f2)) <= tol

    def hybrid_ill_conditioned(x0, tol=TOL):
        """
        Direct approach for ill-conditioned problems:
        1. First try secant (most efficient for smooth functions)
        2. If that fails, try Newton with adaptive damping
        3. If that fails, try bisection with wider search
        4. If all fail, use Nelder-Mead to minimize |f(x)|
        5. Polish the best result
        """
        # Strategy 1: Secant method
        if secant_method(x0):
            if polish(best_x, POLISH_TOL):
                return True
            return True
        
        # Strategy 2: Newton with forward difference
        if newton_forward(x0):
            if polish(best_x, POLISH_TOL):
                return True
            return True
        
        # Strategy 3: Bisection
        if bisection(x0):
            if polish(best_x, POLISH_TOL):
                return True
            return True
        
        # Strategy 4: Nelder-Mead to minimize |f(x)|
        if nelder_mead(x0):
            if polish(best_x, POLISH_TOL):
                return True
            return True
        
        # Final polish attempt on best point found
        if polish(best_x, TOL):
            return True
        
        return False

    # Try the hybrid approach
    hybrid_ill_conditioned(x0)
    
    return best_x
# EVOLVE-BLOCK-END
