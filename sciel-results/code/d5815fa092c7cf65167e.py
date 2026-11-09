"""Root finder for 3 fixed two-dimensional systems using Broyden's method with robust line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with good initial Jacobian approximation.
    2. Multiple starting points if initial fails.
    3. Time-guarded with explicit deadline.
    """
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin before 150s kill
    
    x0 = np.array(x0, dtype=float, copy=True)
    
    def time_ok():
        return time.time() < deadline
    
    def norm(v):
        try:
            v = np.asarray(v, dtype=float)
            if v.shape != (2,) or not np.all(np.isfinite(v)):
                return np.inf
            return float(np.linalg.norm(v))
        except:
            return np.inf
    
    best_x = x0.copy()
    best_norm = np.inf
    
    def update_best(x, fx=None):
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return
            if fx is None:
                fx = F(x)
            n = norm(fx)
            if n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass
    
    # Check initial point
    try:
        fx0 = F(x0)
        n0 = norm(fx0)
        update_best(x0, fx0)
        if n0 <= 1e-8:
            return best_x
    except:
        pass
    
    def broyden_method(x_init, max_iter=100):
        """Broyden's method with damped line search."""
        x = np.array(x_init, dtype=float, copy=True)
        try:
            Fx = np.asarray(F(x), dtype=float)
        except:
            return x
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return x
        fnorm = float(np.linalg.norm(Fx))
        if fnorm <= 1e-8:
            return x
        update_best(x, Fx)
        
        # Compute initial Jacobian using forward differences
        eps = 1e-7
        J = np.zeros((2, 2), dtype=float)
        try:
            for j in range(2):
                e = np.zeros(2)
                e[j] = eps
                Fx_e = np.asarray(F(x + e), dtype=float)
                if Fx_e.shape != (2,) or not np.all(np.isfinite(Fx_e)):
                    # Try smaller eps
                    eps2 = eps * 0.1
                    e[j] = eps2
                    Fx_e = np.asarray(F(x + e), dtype=float)
                    if Fx_e.shape != (2,) or not np.all(np.isfinite(Fx_e)):
                        return x
                    J[:, j] = (Fx_e - Fx) / eps2
                else:
                    J[:, j] = (Fx_e - Fx) / eps
        except:
            return x
        
        # Check if Jacobian is reasonable
        if not np.all(np.isfinite(J)):
            return x
        cond = np.linalg.cond(J)
        if cond > 1e15:
            # Try pseudo-inverse approach or just continue with lstsq
            pass
        
        for _ in range(max_iter):
            if not time_ok():
                break
            
            # Solve J * dx = -Fx
            try:
                if np.linalg.cond(J) > 1e14:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                else:
                    dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Damping: limit step size
            step_norm = float(np.linalg.norm(dx))
            if step_norm > 10.0:
                dx = dx * (10.0 / step_norm)
                step_norm = 10.0
            
            # Backtracking line search with Armijo condition
            alpha = 1.0
            found = False
            for _ in range(30):
                x_new = x + alpha * dx
                if not np.all(np.isfinite(x_new)):
                    alpha *= 0.5
                    continue
                try:
                    Fx_new = np.asarray(F(x_new), dtype=float)
                    if Fx_new.shape != (2,) or not np.all(np.isfinite(Fx_new)):
                        alpha *= 0.5
                        continue
                    n_new = float(np.linalg.norm(Fx_new))
                    # Armijo: n_new < fnorm - c * alpha * ||Fx|| * ||dx||
                    # Simpler: n_new < fnorm * (1 - 1e-4 * alpha)
                    if n_new < fnorm * (1 - 1e-4 * alpha):
                        # Broyden update: J = J + (y - J*s) * s^T / (s^T * s)
                        s = x_new - x
                        y = Fx_new - Fx
                        Js = J @ s
                        denom = float(s @ s)
                        if denom > 1e-16:
                            J = J + ((y - Js) * s.T) / denom
                        
                        x = x_new
                        Fx = Fx_new
                        fnorm = n_new
                        found = True
                        update_best(x, Fx)
                        if fnorm <= 1e-8:
                            return x
                        break
                    else:
                        alpha *= 0.5
                except:
                    alpha *= 0.5
            
            if not found:
                break
        
        return x
    
    def newton_fd(x_init, max_iter=30):
        """Newton with finite difference Jacobian as backup."""
        x = np.array(x_init, dtype=float, copy=True)
        try:
            Fx = np.asarray(F(x), dtype=float)
            if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
                return x
        except:
            return x
        fnorm = float(np.linalg.norm(Fx))
        if fnorm <= 1e-8:
            return x
        update_best(x, Fx)
        
        for _ in range(max_iter):
            if not time_ok():
                break
            
            # Central differences for Jacobian
            eps = 1e-6
            J = np.zeros((2, 2), dtype=float)
            try:
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = eps
                    Fx_p = np.asarray(F(x + e), dtype=float)
                    Fx_m = np.asarray(F(x - e), dtype=float)
                    if Fx_p.shape != (2,) or not np.all(np.isfinite(Fx_p)):
                        break
                    if Fx_m.shape != (2,) or not np.all(np.isfinite(Fx_m)):
                        break
                    J[:, j] = (Fx_p - Fx_m) / (2 * eps)
            except:
                break
            
            if not np.all(np.isfinite(J)):
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            if not np.all(np.isfinite(dx)):
                break
            
            step_norm = float(np.linalg.norm(dx))
            if step_norm > 10.0:
                dx = dx * (10.0 / step_norm)
            
            # Backtracking
            alpha = 1.0
            found = False
            for _ in range(25):
                x_new = x + alpha * dx
                if not np.all(np.isfinite(x_new)):
                    alpha *= 0.5
                    continue
                try:
                    Fx_new = np.asarray(F(x_new), dtype=float)
                    if Fx_new.shape != (2,) or not np.all(np.isfinite(Fx_new)):
                        alpha *= 0.5
                        continue
                    n_new = float(np.linalg.norm(Fx_new))
                    if n_new < fnorm:
                        x = x_new
                        Fx = Fx_new
                        fnorm = n_new
                        found = True
                        update_best(x, Fx)
                        if fnorm <= 1e-8:
                            return x
                        break
                    else:
                        alpha *= 0.5
                except:
                    alpha *= 0.5
            
            if not found:
                break
        
        return x
    
    # Try Broyden from initial point
    try:
        x_result = broyden_method(x0)
        if x_result is not None:
            update_best(x_result)
            if best_norm <= 1e-8:
                return best_x
    except:
        pass
    
    # If not found, try from perturbed points
    scale = max(1.0, float(np.linalg.norm(x0)))
    perturbations = [
        np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]), np.array([0.0, -1.0]),
        np.array([1.0, 1.0]) / np.sqrt(2),
        np.array([-1.0, -1.0]) / np.sqrt(2),
        np.array([1.0, -1.0]) / np.sqrt(2),
        np.array([-1.0, 1.0]) / np.sqrt(2),
    ]
    scales = [0.5, 1.0, 2.0, 5.0]
    
    for s in scales:
        for off in perturbations:
            if not time_ok():
                break
            if best_norm <= 1e-8:
                break
            x_try = x0 + s * scale * off
            try:
                x_result = broyden_method(x_try, max_iter=50)
                if x_result is not None:
                    update_best(x_result)
                    if best_norm <= 1e-8:
                        return best_x
            except:
                pass
    
    # Final fallback: Newton from best
    if best_norm > 1e-8 and best_norm < np.inf and time_ok():
        try:
            x_result = newton_fd(best_x, max_iter=20)
            if x_result is not None:
                update_best(x_result)
        except:
            pass
    
    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
