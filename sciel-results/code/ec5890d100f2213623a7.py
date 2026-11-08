import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return np.all(np.isfinite(x)) and x.shape == (2,)

def _norm(x):
    return float(np.linalg.norm(x))

def _safe_call(F, x):
    """Call F safely, returning None on failure."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None

def find_root_system(F, x0):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 110.0  # Safety margin from 120s limit
    
    x = np.array(x0, dtype=float)
    
    # Initial evaluation
    Fx = _safe_call(F, x)
    if Fx is None:
        # If initial point is invalid, try a small perturbation
        x = x + np.array([0.1, 0.1])
        Fx = _safe_call(F, x)
        if Fx is None:
            return x

    if _norm(Fx) <= 1e-12:
        return x

    # Newton-Raphson with finite difference Jacobian
    # For 2D, we need F(x), F(x+e1), F(x+e2) to build J
    # We already have F(x).
    
    max_iter = 50
    h = 1e-6  # Step size for finite differences
    
    for i in range(max_iter):
        if _elapsed() > deadline:
            break
            
        # Compute Jacobian using forward differences
        # We need F(x + h*e1) and F(x + h*e2)
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        Fx1 = _safe_call(F, x + e1)
        Fx2 = _safe_call(F, x + e2)
        
        if Fx1 is None or Fx2 is None:
            # If we can't evaluate nearby points, we might be at a boundary or singularity.
            # Try to return best x or perturb.
            break
            
        # J = [[dF1/dx1, dF1/dx2], [dF2/dx1, dF2/dx2]]
        J = np.zeros((2, 2))
        J[:, 0] = (Fx1 - Fx) / h
        J[:, 1] = (Fx2 - Fx) / h
        
        # Solve J * d = -Fx
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, try a small perturbation to x and restart
            # or just break if we can't proceed.
            # A simple fallback: use the direction of -Fx if J is singular?
            # Or just break and return current best.
            break
        
        # Check for NaNs in step
        if not np.all(np.isfinite(d)):
            break
            
        # Newton step with backtracking line search
        alpha = 1.0
        norm_current = _norm(Fx)
        found_step = False
        
        for _ls in range(20):
            if _elapsed() > deadline:
                break
                
            x_new = x + alpha * d
            Fx_new = _safe_call(F, x_new)
            
            if Fx_new is None:
                alpha *= 0.5
                continue
                
            norm_new = _norm(Fx_new)
            
            if norm_new < norm_current:
                # Accept step
                x = x_new
                Fx = Fx_new
                found_step = True
                break
            else:
                alpha *= 0.5
                
        if not found_step:
            # If no step could reduce the norm, we are likely stuck or at a minimum.
            # Try a small random perturbation to escape local issues?
            # Or just break.
            # Given the task is root finding, if we can't reduce norm, we might be far from root.
            # Let's try a small perturbation in the direction of -Fx if J was singular or step failed.
            # Actually, if line search failed completely, it's better to break and return best.
            break
            
        if _norm(Fx) <= 1e-12:
            break
            
        # Optional: If progress is extremely slow, consider restarting with a different x?
        # For now, standard Newton should handle most 2D systems well.
        
    return x
