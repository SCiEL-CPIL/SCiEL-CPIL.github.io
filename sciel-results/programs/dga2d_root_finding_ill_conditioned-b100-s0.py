"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import brentq, newton, minimize_scalar

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Uses a stochastic global optimization approach:
    1. Multi-start random restarts to find a bracket or close point.
    2. Safeguarded Inverse Quadratic Interpolation (SIQI) for local convergence.
    3. Derivative-free minimization of f^2 for non-sign-changing roots.
    """
    x0 = float(x0)
    
    # State
    eval_count = 0
    max_evals = 500
    
    # Wrapped function to count evaluations
    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        return f(x)

    best_x = x0
    best_f = abs(f_counted(x0))
    if best_f <= 1e-8:
        return best_x

    # Safeguarded Secant/Bisection Solver (approximating SIQI)
    def siqi_solve(a, b, fa, fb, max_iter=50, tol=1e-12):
        nonlocal eval_count, best_x, best_f
        
        if fa == 0: return a
        if fb == 0: return b
        if fa * fb > 0:
            return None

        c = a
        fc = fa
        d = b - a
        e = d
        
        for _ in range(max_iter):
            if eval_count >= max_evals - 5:
                break
            
            # Ensure |fa| < |fb|
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
                d = b - a
                e = d
            
            if abs(fa) <= 1e-12:
                return a
            if abs(d) < 1e-12:
                return a

            x_try = None
            
            # Try Secant step (first-order IQI)
            try:
                denom = fb - fa
                if abs(denom) > 1e-300:
                    x_secant = b - fb * (b - a) / denom
                    mid = (a + b) / 2.0
                    if a < x_secant < b:
                        # Check if Secant step is "good" (not too close to midpoint)
                        if abs(x_secant - mid) > 0.1 * abs(mid - a):
                            x_try = x_secant
                        else:
                            x_try = mid
                    else:
                        x_try = mid
                else:
                    x_try = (a + b) / 2.0
            except Exception:
                x_try = (a + b) / 2.0
            
            fx_try = f_counted(x_try)
            
            if fx_try == 0:
                return x_try
            if abs(fx_try) < best_f:
                best_x = x_try
                best_f = abs(fx_try)
                
            if fa * fx_try < 0:
                b = x_try
                fb = fx_try
            else:
                c = a
                fc = fa
                a = x_try
                fa = fx_try
                
            if abs(fb) <= 1e-12 or abs(b - a) < 1e-12:
                return b if abs(fb) < abs(fa) else a

        return b if abs(fb) < abs(fa) else a

    # Multi-Start Strategy
    def attempt_solve_from_point(x_start):
        nonlocal eval_count, best_x, best_f
        
        if eval_count >= max_evals - 10:
            return False
            
        # Try to find a bracket around x_start
        scales = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
        
        # First, check x_start itself
        fx_start = f_counted(x_start)
        if abs(fx_start) < best_f:
            best_x = x_start
            best_f = abs(fx_start)
        if best_f <= 1e-8:
            return True
            
        for s in scales:
            if eval_count >= max_evals - 10:
                break
            a = x_start - s
            b = x_start + s
            
            fa = f_counted(a)
            fb = f_counted(b)
            
            # Update best
            if abs(fa) < best_f:
                best_x = a
                best_f = abs(fa)
            if abs(fb) < best_f:
                best_x = b
                best_f = abs(fb)
                
            if best_f <= 1e-8:
                return True
                
            # Check for sign change
            if fa * fb < 0:
                # Found a bracket! Use SIQI
                root = siqi_solve(a, b, fa, fb)
                if root is not None:
                    f_root = f_counted(root) # Verify
                    if abs(f_root) <= 1e-8:
                        return True
                    if abs(f_root) < best_f:
                        best_x = root
                        best_f = abs(f_root)
                return False # Break out of scale loop if we found a bracket and tried it
            
            # If no sign change, continue to larger scale

        return False

    # Phase 1: Try local expansion from x0
    if attempt_solve_from_point(x0):
        return best_x

    # Phase 2: Random Restart Strategy
    np.random.seed(42)
    
    # Generate a set of candidate points
    for radius in [0.1, 0.5, 1, 5, 10, 50, 100, 500]:
        if eval_count >= max_evals - 10:
            break
            
        # Try a few random points at this radius
        for _ in range(5):
            if eval_count >= max_evals - 10:
                break
            
            angle = np.random.uniform(0, 2 * np.pi)
            x_try = x0 + radius * np.cos(angle)
            
            if attempt_solve_from_point(x_try):
                return best_x

    # Phase 3: If we haven't found a sign change, minimize f(x)^2
    if eval_count < max_evals - 20:
        try:
            bound = max(1.0, abs(best_x)) * 10
            res = minimize_scalar(lambda x: f_counted(x)**2, bounds=(best_x - bound, best_x + bound), 
                                   method='bounded', options={'maxiter': 50, 'xatol': 1e-12})
            
            if res.success:
                f_res = f_counted(res.x)
                if abs(f_res) < best_f:
                    best_x = res.x
                    best_f = abs(f_res)
                    
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
