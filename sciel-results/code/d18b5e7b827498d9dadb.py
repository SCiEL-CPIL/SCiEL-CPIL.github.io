import numpy as np
import time
from scipy.optimize import root, minimize

def _broyden_refine(F, x, J, max_iter=30):
    """
    Refine x using Broyden's method with the provided Jacobian.
    Returns (x, J, converged) where converged is True if ||F(x)|| <= 1e-8.
    """
    x = np.array(x, dtype=float).copy()
    J = np.array(J, dtype=float).copy()
    
    try:
        Fx = F(x)
    except Exception:
        return x, J, False
    
    if not np.all(np.isfinite(Fx)):
        return x, J, False
    
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x, J, True
    
    for _ in range(max_iter):
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        # Simple line search
        alpha = 1.0
        step_accepted = False
        for _ in range(10):
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
                return x_new, J, True
            
            if norm_Fx_new < (1.0 - 1e-4 * alpha) * norm_Fx:
                # Broyden rank-1 update
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
    
    return x, J, norm_Fx <= 1e-8


def _compute_jacobian(F, x, h=1e-5):
    """Compute Jacobian using central differences."""
    J = np.zeros((2, 2))
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
                Fx = F(x)
                if np.all(np.isfinite(F_plus)):
                    J[:, j] = (F_plus - Fx) / h
                elif np.all(np.isfinite(F_minus)):
                    J[:, j] = (Fx - F_minus) / h
                else:
                    J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
        except Exception:
            J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
    return J


def _homotopy_fallback(F, x0, deadline):
    """
    Homotopy/continuation method: solve H(x, t) = (1-t)*x + t*F(x) = 0
    by marching t from 0 to 1. At t=0, the solution is x=0.
    """
    def time_up():
        return time.time() > deadline
    
    # Start from t=0 where solution is x=0
    x = np.zeros(2)
    t = 0.0
    
    # Initial step size for t
    dt = 0.05
    max_t_steps = 50
    max_inner_iters = 15
    
    for t_step in range(max_t_steps):
        if time_up():
            break
        
        t_new = min(t + dt, 1.0)
        
        # Define homotopy function at current t
        def H(x_val):
            return (1 - t_new) * x_val + t_new * F(x_val)
        
        # Use Broyden to solve H(x) = 0 starting from current x
        try:
            Hx = H(x)
        except Exception:
            dt *= 0.5
            continue
        
        if not np.all(np.isfinite(Hx)):
            dt *= 0.5
            continue
        
        # Compute Jacobian of H: dH/dx = (1-t)*I + t*J_F
        J_F = _compute_jacobian(F, x)
        J_H = (1 - t_new) * np.eye(2) + t_new * J_F
        
        # Try to solve using Broyden with J_H as initial Jacobian
        x, J_H, converged = _broyden_refine(H, x, J_H, max_iter=max_inner_iters)
        
        if converged:
            t = t_new
            # Increase step size if we converged easily
            dt = min(dt * 1.5, 0.2)
        else:
            # Reduce step size
            dt *= 0.5
            if dt < 1e-3:
                # Try a smaller step
                dt = 1e-3
        
        if t >= 1.0:
            break
    
    # Check if we found a root of the original F
    try:
        Fx = F(x)
        if np.all(np.isfinite(Fx)) and np.linalg.norm(Fx) <= 1e-8:
            return x
    except Exception:
        pass
    
    # If homotopy didn't fully converge, try to polish from the best point
    try:
        Fx = F(x)
        if np.all(np.isfinite(Fx)):
            J_F = _compute_jacobian(F, x)
            x, _, converged = _broyden_refine(F, x, J_F, max_iter=20)
            if converged:
                return x
            # Check again
            Fx = F(x)
            if np.all(np.isfinite(Fx)) and np.linalg.norm(Fx) <= 1e-8:
                return x
    except Exception:
        pass
    
    return x


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Strategy: Broyden from x0, Nelder-Mead global search, homotopy, final polish.
    """
    x0_arr = np.array(x0, dtype=float).copy()
    if x0_arr.shape != (2,):
        x0_arr = x0_arr.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    def objective(x):
        """Scalar objective: ||F(x)||^2"""
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return 1e10
            return float(np.dot(Fx, Fx))
        except Exception:
            return 1e10
    
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
    
    # Strategy 1: Broyden's method from x0
    if not time_up():
        try:
            J = _compute_jacobian(F, x0_arr)
            x, J, converged = _broyden_refine(F, x0_arr, J, max_iter=30)
            if converged:
                return x
            # Update best
            try:
                Fx = F(x)
                if np.all(np.isfinite(Fx)):
                    norm_Fx = np.linalg.norm(Fx)
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x.copy()
            except Exception:
                pass
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead global search to escape local minima
    if not time_up():
        try:
            # Use multiple starting points for robustness
            start_points = [best_x.copy()]
            
            # Add perturbed versions of x0
            for scale in [0.1, 1.0, 10.0]:
                for dx in [-1, 1]:
                    for dy in [-1, 1]:
                        p = x0_arr + np.array([dx * scale, dy * scale])
                        start_points.append(p)
            
            # Also try origin and other common points
            start_points.append(np.zeros(2))
            start_points.append(np.array([1.0, 1.0]))
            start_points.append(np.array([-1.0, -1.0]))
            
            nm_best_x = best_x.copy()
            nm_best_norm = best_norm
            
            for sp in start_points:
                if time_up():
                    break
                try:
                    res = minimize(objective, sp, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-16})
                    if res.success or res.fun < nm_best_norm:
                        Fx = F(res.x)
                        if np.all(np.isfinite(Fx)):
                            norm_Fx = np.linalg.norm(Fx)
                            if norm_Fx < nm_best_norm:
                                nm_best_norm = norm_Fx
                                nm_best_x = res.x.copy()
                                if norm_Fx <= 1e-8:
                                    best_x = res.x.copy()
                                    best_norm = norm_Fx
                                    break
                except Exception:
                    continue
            
            # Polish the Nelder-Mead result with Broyden
            if nm_best_norm < best_norm:
                try:
                    J = _compute_jacobian(F, nm_best_x)
                    x, J, converged = _broyden_refine(F, nm_best_x, J, max_iter=20)
                    if converged:
                        return x
                    Fx = F(x)
                    if np.all(np.isfinite(Fx)):
                        norm_Fx = np.linalg.norm(Fx)
                        if norm_Fx < best_norm:
                            best_norm = norm_Fx
                            best_x = x.copy()
                except Exception:
                    pass
        except Exception:
            pass
    
    # Strategy 3: Homotopy/continuation method
    if not time_up():
        try:
            x_homo = _homotopy_fallback(F, x0_arr, deadline)
            try:
                Fx = F(x_homo)
                if np.all(np.isfinite(Fx)) and np.linalg.norm(Fx) <= 1e-8:
                    return x_homo
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x_homo.copy()
            except Exception:
                pass
        except Exception:
            pass
    
    # Strategy 4: Final Broyden polish from best point
    if not time_up() and best_norm > 1e-8:
        try:
            J = _compute_jacobian(F, best_x)
            x, J, converged = _broyden_refine(F, best_x, J, max_iter=30)
            if converged:
                return x
            Fx = F(x)
            if np.all(np.isfinite(Fx)):
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x.copy()
        except Exception:
            pass
    
    # Strategy 5: SciPy hybrid from best point as last resort
    if not time_up() and best_norm > 1e-8:
        try:
            res = root(F, best_x, method='hybr', options={'maxfev': 100})
            if res.success:
                return res.x
            if np.linalg.norm(res.fun) <= 1e-8:
                return res.x
            if np.linalg.norm(res.fun) < best_norm:
                best_norm = np.linalg.norm(res.fun)
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x
