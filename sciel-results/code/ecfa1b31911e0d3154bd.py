"""Combined root finder: global Nelder-Mead screening + Newton polish for 3 fixed 2D systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# We track time to avoid exceeding the 150s limit (with margin).
# The evaluator kills after 150s, but we should finish well before.
# We'll use a deadline guard with safety margin.

_START_TIME = None
_DEADLINE_MARGIN = 10  # seconds to spare before 150s limit

def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0

def _time_left():
    return 150.0 - _elapsed() - _DEADLINE_MARGIN

def _safe_norm(F, x):
    """Safely compute ||F(x)||, returning inf on error."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return np.inf
        return float(np.linalg.norm(Fx))
    except Exception:
        return np.inf

def _newton_polish(F, x, max_iter=50, tol=1e-12):
    """
    Newton's method with finite-difference Jacobian.
    Uses central differences for better accuracy.
    Returns (x, norm_F, success).
    """
    x = np.array(x, dtype=float)
    for _ in range(max_iter):
        if _time_left() < 0.5:
            return x, _safe_norm(F, x), False
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
                return x, np.inf, False
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= tol:
                return x, norm_Fx, True
            
            # Compute Jacobian using central differences for better accuracy
            h = 1e-7
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_plus = F(x + e)
                Fx_minus = F(x - e)
                Fx_plus = np.asarray(Fx_plus, dtype=float)
                Fx_minus = np.asarray(Fx_minus, dtype=float)
                
                if np.all(np.isfinite(Fx_plus)) and np.all(np.isfinite(Fx_minus)):
                    J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
                else:
                    # Fall back to forward difference if central fails
                    if not np.all(np.isfinite(Fx_plus)):
                        return x, norm_Fx, False
                    J[:, j] = (Fx_plus - Fx) / h
            
            try:
                dx = np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                return x, np.inf, False
                
            x_new = x - dx
            if not np.all(np.isfinite(x_new)):
                return x, norm_Fx, False
            x = x_new
        except Exception:
            return x, _safe_norm(F, x), False
    
    norm_final = _safe_norm(F, x)
    return x, norm_final, norm_final <= tol

def _nelder_mead(F, x0, max_iter=200):
    """
    Run Nelder-Mead on ||F(x)||^2.
    Returns (x, norm_F).
    """
    def objective(x):
        try:
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return 1e100
            return float(np.dot(fx, fx))
        except Exception:
            return 1e100

    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-20, 'maxiter': max_iter, 'maxfev': 300})
        x_opt = res.x
        norm_opt = _safe_norm(F, x_opt)
        return x_opt, norm_opt
    except Exception:
        return x0, _safe_norm(F, x0)

def find_root_system(F, x0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    x0 = np.asarray(x0, dtype=float).flatten()
    if x0.shape != (2,):
        x0 = np.array([0.0, 0.0])
    
    best_x = x0
    best_norm = np.inf
    
    def _update_best(x_candidate):
        nonlocal best_x, best_norm
        if x_candidate is None:
            return
        try:
            x_candidate = np.asarray(x_candidate, dtype=float).flatten()
            if x_candidate.shape != (2,) or not np.all(np.isfinite(x_candidate)):
                return
            n = _safe_norm(F, x_candidate)
            if n < best_norm:
                best_norm = n
                best_x = x_candidate
                if n <= 1e-8:
                    return  # We found a valid root
        except Exception:
            pass
    
    # Strategy 1: Check x0
    _update_best(x0)
    if best_norm <= 1e-8:
        return best_x
    
    # Strategy 2: Newton polish from x0
    x_res, n_res, _ = _newton_polish(F, x0, max_iter=50)
    _update_best(x_res)
    if best_norm <= 1e-8:
        return best_x
    
    # Strategy 3: Nelder-Mead from x0
    if _time_left() > 2.0:
        x_nm, n_nm = _nelder_mead(F, x0, max_iter=200)
        _update_best(x_nm)
        if best_norm <= 1e-8:
            return best_x
        
        # Newton polish from Nelder-Mead result
        if _time_left() > 1.0:
            x_res, n_res, _ = _newton_polish(F, x_nm, max_iter=50)
            _update_best(x_res)
            if best_norm <= 1e-8:
                return best_x
    
    # Strategy 4: Multi-start with deterministic grid + Newton
    # Generate a set of initial points around x0
    initial_points = [x0]
    # Add points in a circle around x0 at various radii
    for r in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            x_try = x0 + np.array([dx, dy])
            initial_points.append(x_try)
    
    # Screen initial points
    screened_points = []
    for x_try in initial_points:
        if _time_left() < 1.0:
            break
        norm_try = _safe_norm(F, x_try)
        screened_points.append((norm_try, x_try))
        if norm_try < best_norm:
            best_norm = norm_try
            best_x = x_try
        if norm_try <= 1e-8:
            return x_try
    
    # Sort by norm
    screened_points.sort(key=lambda t: t[0])
    
    # Try Newton from the top 5 best points
    for norm_try, x_try in screened_points[:5]:
        if _time_left() < 1.0:
            break
        x_new, norm_new, success = _newton_polish(F, x_try, max_iter=50, tol=1e-12)
        _update_best(x_new)
        if best_norm <= 1e-8:
            return best_x
    
    # Strategy 5: Nelder-Mead from the best screened point
    if _time_left() > 2.0:
        x_nm, n_nm = _nelder_mead(F, best_x, max_iter=200)
        _update_best(x_nm)
        if best_norm <= 1e-8:
            return best_x
        
        # Newton polish from Nelder-Mead result
        if _time_left() > 1.0:
            x_res, n_res, _ = _newton_polish(F, x_nm, max_iter=50)
            _update_best(x_res)
            if best_norm <= 1e-8:
                return best_x
    
    # Strategy 6: More random starts with Nelder-Mead
    rng = np.random.default_rng(42)
    for i in range(10):
        if _time_left() < 1.0:
            break
        offset = rng.uniform(-50.0, 50.0, size=2)
        x_init = x0 + offset
        x_nm, n_nm = _nelder_mead(F, x_init, max_iter=100)
        _update_best(x_nm)
        if best_norm <= 1e-8:
            return best_x
        
        if _time_left() < 1.0:
            break
        x_res, n_res, _ = _newton_polish(F, x_nm, max_iter=30)
        _update_best(x_res)
        if best_norm <= 1e-8:
            return best_x
    
    # Return best found
    return best_x

# EVOLVE-BLOCK-END
