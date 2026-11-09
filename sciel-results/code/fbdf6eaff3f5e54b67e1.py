"""Root finder for 3 fixed two-dimensional systems using a hybrid local-global approach."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


def _safe_F(F, x):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(x)
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _local_broyden(F, x0, max_iter=20):
    """
    Runs Broyden's method from x0 for a limited number of iterations.
    Returns (x_best, best_norm, calls_used).
    """
    x = np.array(x0, dtype=float, copy=True)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
        
    calls = 0
    Fx = _safe_F(F, x)
    calls += 1
    if Fx is None:
        return x, np.inf, calls
        
    best_norm = np.linalg.norm(Fx)
    x_best = x.copy()
    
    if best_norm <= 1e-8:
        return x_best, best_norm, calls

    # Build initial Jacobian
    h = 1e-5
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fp = _safe_F(F, x + e)
        calls += 1
        if Fp is not None:
            J[:, j] = (Fp - Fx) / h
        else:
            J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])

    if np.linalg.det(J) < 1e-12:
        J += 1e-8 * np.eye(2)

    for _ in range(max_iter):
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
            
        # Simple line search
        alpha = 1.0
        x_trial = x + alpha * dx
        Fx_trial = _safe_F(F, x_trial)
        calls += 1
        
        if Fx_trial is None:
            break
            
        trial_norm = np.linalg.norm(Fx_trial)
        
        # Backtracking
        while trial_norm > 0.9 * best_norm and alpha > 1e-8:
            alpha *= 0.5
            x_trial = x + alpha * dx
            Fx_trial = _safe_F(F, x_trial)
            calls += 1
            if Fx_trial is None:
                break
            if Fx_trial is not None:
                trial_norm = np.linalg.norm(Fx_trial)
            else:
                break
                
        if Fx_trial is None:
            break
            
        # Update Jacobian (Broyden)
        s = x_trial - x
        y = Fx_trial - Fx
        denom = np.dot(y, s)
        if abs(denom) > 1e-14:
            J = J + np.outer(y - J @ s, s) / denom
            
        x = x_trial
        Fx = Fx_trial
        best_norm = trial_norm
        x_best = x.copy()
        
        if best_norm <= 1e-8:
            return x_best, best_norm, calls
            
    return x_best, best_norm, calls


def find_root_system(F, x0):
    # Strategy 1: Try local Broyden from several starting points
    starts = [
        np.array(x0, dtype=float),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([-1.0, -1.0]),
        np.array([1.0, -1.0]),
        np.array([-1.0, 1.0])
    ]
    
    best_x = None
    best_norm = np.inf
    total_calls = 0
    
    for s in starts:
        if not np.all(np.isfinite(s)):
            continue
        x_cand, norm_cand, calls_cand = _local_broyden(F, s, max_iter=10)
        total_calls += calls_cand
        if norm_cand < best_norm:
            best_norm = norm_cand
            best_x = x_cand
        if best_norm <= 1e-8:
            return best_x
            
    # Strategy 2: Global search with Differential Evolution
    # We restrict iterations to keep call count low (< 500 total)
    # Current total_calls is likely around 6 * (1 + 2 + 10*2) ~ 120-150 calls
    # DE with popsize=10, maxiter=20 will add ~200 calls. Total ~350.
    
    try:
        def objective(x):
            Fx = _safe_F(F, x)
            if Fx is None:
                return 1e10
            return np.dot(Fx, Fx)
            
        bounds = [(-100.0, 100.0), (-100.0, 100.0)]
        # maxiter and popsize control the number of function evaluations
        result = differential_evolution(objective, bounds, maxiter=20, popsize=10, seed=42)
        
        if result.success and np.all(np.isfinite(result.x)):
            # Polish the result with local Broyden
            x_polished, norm_polished, calls_polished = _local_broyden(F, result.x, max_iter=20)
            total_calls += calls_polished
            
            if norm_polished < best_norm:
                best_norm = norm_polished
                best_x = x_polished
    except Exception:
        pass
        
    return best_x if best_x is not None else np.array(x0, dtype=float)
# EVOLVE-BLOCK-END
