"""Root finder for 3 fixed 2D systems using a robust multi-strategy approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a combination of Broyden's method with adaptive Jacobian updates,
    multiple restart strategies, and SciPy fallback.
    """
    x0_arr = np.array(x0, dtype=float).copy()
    if x0_arr.shape != (2,):
        x0_arr = x0_arr.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    best_x = x0_arr.copy()
    best_norm = np.inf
    
    # Try to get initial evaluation
    try:
        Fx = F(x0_arr)
        if np.all(np.isfinite(Fx)):
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x0_arr.copy()
            if norm_Fx <= 1e-8:
                return x0_arr
    except Exception:
        pass
    
    # Strategy 1: Broyden's method with good initial Jacobian
    def try_broyden(x_start, Fx_start, max_iter=40):
        nonlocal best_x, best_norm, time_up, deadline
        
        if not np.all(np.isfinite(Fx_start)):
            return None
        
        norm_Fx = np.linalg.norm(Fx_start)
        if norm_Fx <= 1e-8:
            return x_start
        
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_start.copy()
        
        x = x_start.copy()
        Fx = Fx_start.copy()
        
        # Compute initial Jacobian using central differences for better accuracy
        J = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            try:
                F_plus = F(x + e)
                F_minus = F(x - e)
                if np.all(np.isfinite(F_plus)) and np.all(np.isfinite(F_minus)):
                    J[:, j] = (F_plus - F_minus) / (2 * h)
                else:
                    # Fallback to forward difference
                    if np.all(np.isfinite(F_plus)):
                        J[:, j] = (F_plus - Fx) / h
                    else:
                        J[:, j] = (Fx - F_minus) / h
            except Exception:
                J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
        
        # Check if Jacobian is well-conditioned
        try:
            cond = np.linalg.cond(J)
            if cond > 1e12:
                # Regularize
                J += 1e-10 * np.eye(2)
        except Exception:
            J = np.eye(2)
        
        for _ in range(max_iter):
            if time_up():
                break
            
            try:
                d = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            # Adaptive line search
            alpha = 1.0
            step_accepted = False
            for _ in range(12):
                if time_up():
                    break
                
                x_new = x + alpha * d
                try:
                    Fx_new = F(x_new)
                except Exception:
                    alpha *= 0.5
                    continue
                
                if not np.all(np.isfinite(Fx_new)):
                    alpha *= 0.5
                    continue
                
                norm_Fx_new = np.linalg.norm(Fx_new)
                
                if norm_Fx_new <= 1e-8:
                    if norm_Fx_new < best_norm:
                        best_norm = norm_Fx_new
                        best_x = x_new.copy()
                    return x_new
                
                if norm_Fx_new < best_norm:
                    best_norm = norm_Fx_new
                    best_x = x_new.copy()
                
                # Armijo condition
                if norm_Fx_new < (1.0 - 1e-4 * alpha) * norm_Fx:
                    # Broyden update
                    d_dot_d = np.dot(d, d)
                    if d_dot_d > 1e-14:
                        delta_F = Fx_new - Fx - J @ d
                        J = J + np.outer(delta_F, d) / d_dot_d
                    
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
                    step_accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not step_accepted:
                break
        
        return x if norm_Fx < best_norm else None
    
    # Try Broyden from x0
    try:
        Fx0 = F(x0_arr)
        if np.all(np.isfinite(Fx0)):
            result = try_broyden(x0_arr, Fx0)
            if result is not None and np.linalg.norm(F(result)) <= 1e-8:
                return result
    except Exception:
        pass
    
    # Strategy 2: Try multiple random restarts with small perturbations
    rng = np.random.default_rng(42)
    for attempt in range(8):
        if time_up():
            break
        
        # Perturb x0
        scale = 0.1 * (attempt + 1)
        perturbation = rng.normal(0, scale, size=2)
        x_try = x0_arr + perturbation
        
        try:
            Fx_try = F(x_try)
            if not np.all(np.isfinite(Fx_try)):
                continue
            
            result = try_broyden(x_try, Fx_try, max_iter=25)
            if result is not None:
                try:
                    if np.linalg.norm(F(result)) <= 1e-8:
                        return result
                except Exception:
                    pass
        except Exception:
            continue
    
    # Strategy 3: SciPy's hybrid method from best point found
    try:
        res = root(F, best_x, method='hybr', options={'maxfev': 500})
        if res.success:
            return res.x
        if np.linalg.norm(res.fun) <= 1e-8:
            return res.x
        if np.linalg.norm(res.fun) < best_norm:
            best_x = res.x
            best_norm = np.linalg.norm(res.fun)
    except Exception:
        pass
    
    # Strategy 4: SciPy's lm method (Levenberg-Marquardt)
    try:
        res = root(F, best_x, method='lm', options={'maxfev': 500})
        if res.success:
            return res.x
        if np.linalg.norm(res.fun) <= 1e-8:
            return res.x
        if np.linalg.norm(res.fun) < best_norm:
            best_x = res.x
            best_norm = np.linalg.norm(res.fun)
    except Exception:
        pass
    
    # Strategy 5: Try SciPy from original x0 if we haven't tried it
    try:
        res = root(F, x0_arr, method='hybr', options={'maxfev': 300})
        if res.success:
            return res.x
        if np.linalg.norm(res.fun) <= 1e-8:
            return res.x
        if np.linalg.norm(res.fun) < best_norm:
            best_x = res.x
            best_norm = np.linalg.norm(res.fun)
    except Exception:
        pass
    
    return best_x

# EVOLVE-BLOCK-END
