"""Root finder for 3 fixed two-dimensional systems using Multi-Start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE

def _is_finite(arr):
    try:
        arr = np.asarray(arr, dtype=float)
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False

def _norm_sq(x, F):
    """Compute squared Euclidean norm of F(x). Returns inf if invalid."""
    try:
        Fx = F(x)
        if not _is_finite(Fx):
            return np.inf
        return np.dot(Fx, Fx)
    except Exception:
        return np.inf

def _nelder_mead(F, x0, max_iter=50, tol=1e-14):
    """
    Simple Nelder-Mead implementation for 2D minimization of ||F(x)||^2.
    Returns (x_best, best_norm_sq, calls_used).
    """
    x = np.array(x0, dtype=float)
    # Initial simplex: x, x + delta*e1, x + delta*e2
    delta = 0.5
    simplex = [
        x,
        x + np.array([delta, 0.0]),
        x + np.array([0.0, delta])
    ]
    
    f_vals = [ _norm_sq(s, F) for s in simplex ]
    calls = 3
    
    # Sort simplex by function value
    indices = np.argsort(f_vals)
    simplex = [simplex[i] for i in indices]
    f_vals = [f_vals[i] for i in indices]
    
    best_x = simplex[0]
    best_f = f_vals[0]
    
    if best_f <= tol:
        return best_x, best_f, calls

    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5   # Contraction
    sigma = 0.5 # Shrink
    
    for _ in range(max_iter):
        if not _time_ok():
            break
            
        # Check convergence
        if np.max(f_vals) - np.min(f_vals) < 1e-10:
            break
        if f_vals[0] <= tol:
            break
            
        # Current worst point
        x_worst = simplex[2]
        f_worst = f_vals[2]
        
        # Centroid of best 2 points
        x_centroid = (simplex[0] + simplex[1]) / 2.0
        
        # Reflection
        x_ref = x_centroid + alpha * (x_centroid - x_worst)
        f_ref = _norm_sq(x_ref, F)
        calls += 1
        
        if f_ref < f_vals[0]:
            # Expansion
            x_exp = x_centroid + gamma * (x_centroid - x_worst)
            f_exp = _norm_sq(x_exp, F)
            calls += 1
            if f_exp < f_ref:
                simplex[2] = x_exp
                f_vals[2] = f_exp
            else:
                simplex[2] = x_ref
                f_vals[2] = f_ref
        elif f_ref < f_worst:
            # Accept reflection
            simplex[2] = x_ref
            f_vals[2] = f_ref
        else:
            # Contraction
            x_con = x_centroid + rho * (x_worst - x_centroid)
            f_con = _norm_sq(x_con, F)
            calls += 1
            if f_con < f_worst:
                simplex[2] = x_con
                f_vals[2] = f_con
            else:
                # Shrink
                for i in range(1, 3):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_vals[i] = _norm_sq(simplex[i], F)
                    calls += 1
        
        # Re-sort
        indices = np.argsort(f_vals)
        simplex = [simplex[i] for i in indices]
        f_vals = [f_vals[i] for i in indices]
        
        if f_vals[0] < best_f:
            best_f = f_vals[0]
            best_x = simplex[0].copy()
            
    return best_x, best_f, calls

def find_root_system(F, x0):
    """
    Find root of F(x)=0 using Multi-Start Nelder-Mead.
    """
    x0 = np.array(x0, dtype=float)
    
    # 1. Try local optimization from x0 first
    x_best, f_best, calls_used = _nelder_mead(F, x0, max_iter=80, tol=1e-16)
    
    if f_best <= 1e-16:  # 1e-8 norm -> 1e-16 sq norm
        return x_best
        
    # 2. If local search failed, try global random restarts
    # We don't know the scale of the problem, so we try a few scales around x0
    # and also some random points in a larger box.
    
    # Heuristic: If x0 is close to 0, try scales 1, 10, 100. If x0 is large, try relative scales.
    scale0 = np.linalg.norm(x0)
    if scale0 < 1e-6:
        scales = [1.0, 10.0, 100.0, 1000.0]
    else:
        scales = [scale0 * 0.1, scale0 * 1.0, scale0 * 10.0]
        
    num_restarts_per_scale = 5
    max_total_calls = 400 # Keep total calls reasonable to score well
    
    for scale in scales:
        if calls_used + num_restarts_per_scale * 10 > max_total_calls:
            break
        if not _time_ok():
            break
            
        for _ in range(num_restarts_per_scale):
            if not _time_ok():
                break
            if calls_used >= max_total_calls:
                break
                
            # Random point in box [-scale, scale]
            x_rand = np.random.uniform(-scale, scale, size=2)
            
            # Run a limited Nelder-Mead from this random point
            x_trial, f_trial, calls_trial = _nelder_mead(F, x_rand, max_iter=30, tol=1e-16)
            calls_used += calls_trial
            
            if f_trial < f_best:
                f_best = f_trial
                x_best = x_trial
                if f_best <= 1e-16:
                    return x_best
                    
    # 3. If still not good enough, try one more long run from the best point found so far
    if f_best > 1e-16 and _time_ok():
        x_best, f_best, calls_extra = _nelder_mead(F, x_best, max_iter=100, tol=1e-16)
        calls_used += calls_extra
        
    return x_best
# EVOLVE-BLOCK-END
