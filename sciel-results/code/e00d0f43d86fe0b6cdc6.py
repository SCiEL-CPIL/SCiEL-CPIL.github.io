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
    Fx = _safe_call(F, x)
    if Fx is None:
        # Try a small perturbation if initial point is bad
        x = x + np.array([0.1, 0.1])
        Fx = _safe_call(F, x)
        if Fx is None:
            return x
    
    if _norm(Fx) <= 1e-12:
        return x

    # Initialize Jacobian with finite differences
    J = np.zeros((2, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _safe_call(F, x + e)
        if Fx_plus is None:
            # Fallback: simple secant or just return
            return x
        J[:, j] = (Fx_plus - Fx) / h

    # Broyden's method
    for _ in range(100):
        if _elapsed() > deadline:
            break
            
        # Solve J * d = -Fx
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, try a small perturbation or restart
            # For now, break and return best x
            break

        # Line search
        alpha = 1.0
        Fx_current = Fx
        norm_current = _norm(Fx_current)
        
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
                # Update Broyden
                delta_x = alpha * d
                delta_F = Fx_new - Fx_current
                # Rank-1 update
                if np.dot(delta_x, delta_x) > 1e-16:
                    J += np.outer(delta_F - J @ delta_x, delta_x) / np.dot(delta_x, delta_x)
                
                x = x_new
                Fx = Fx_new
                break
            else:
                alpha *= 0.5
        
        if _norm(Fx) <= 1e-12:
            return x
            
        # If line search failed to reduce norm, we might be stuck.
        # Check if we made any progress in the last few steps.
        # For simplicity, if alpha is very small, break.
        if alpha < 1e-12:
            break
            
    return x
