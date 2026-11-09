"""Root finder for 3 fixed two-dimensional systems using Broyden + damped Newton."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method (cheap Jacobian updates) as primary.
    2. Damped Newton with line search as backup.
    3. Multiple structured initial points.
    4. Time-guarded with explicit deadline.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
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
    
    # Broyden's method (good for minimizing function calls)
    def broyden_method(x_init, max_iter=80):
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
        
        # Compute initial Jacobian (forward differences: 3 calls total)
        eps = 1e-6
        J = np.zeros((2, 2), dtype=float)
        try:
            Fx0 = Fx.copy()
            for j in range(2):
                e = np.zeros(2)
                e[j] = eps
                Fx_e = np.asarray(F(x + e), dtype=float)
                if Fx_e.shape != (2,) or not np.all(np.isfinite(Fx_e)):
                    break
                J[:, j] = (Fx_e - Fx0) / eps
        except:
            return x
        
        # Check if Jacobian is reasonable
        if not np.all(np.isfinite(J)) or np.linalg.cond(J) > 1e12:
            return x
        
        for _ in range(max_iter):
            if not time_ok():
                break
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            # Damping: limit step size
            step_norm = float(np.linalg.norm(dx))
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
                step_norm = 5.0
            
            # Backtracking line search
            alpha = 1.0
            found = False
            for _ in range(20):
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
                        # Broyden update
                        s = x_new - x
                        y = Fx_new - Fx
                        # J = J + (y - J*s) * s^T / (s^T * s)
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
    
    # Damped Newton with line search
    def damped_newton(x_init, max_iter=50):
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
            
            # Compute Jacobian (central differences: 5 calls)
            eps = 1e-7
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
                    if n_new < fnorm * (1 - 1e-4 * alpha * step_norm):
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
    
    # Generate structured initial points
    scale = max(1.0, float(np.linalg.norm(x0)))
    initial_points = [x0.copy()]
    
    # Deterministic perturbations
    offsets = [
        np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]), np.array([0.0, -1.0]),
        np.array([1.0, 1.0]) / np.sqrt(2), np.array([-1.0, -1.0]) / np.sqrt(2),
        np.array([1.0, -1.0]) / np.sqrt(2), np.array([-1.0, 1.0]) / np.sqrt(2),
    ]
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for s in scales:
        for off in offsets:
            initial_points.append(x0 + s * scale * off)
    
    # Try Broyden from each initial point
    for x_init in initial_points:
        if not time_ok():
            break
        if best_norm <= 1e-8:
            break
        
        try:
            x_result = broyden_method(x_init)
            if x_result is not None:
                update_best(x_result)
                if best_norm <= 1e-8:
                    return best_x
        except:
            pass
    
    # If not found, try damped Newton from best and top few points
    if best_norm > 1e-8 and time_ok():
        # Sort initial points by current best norm proxy
        for x_init in initial_points[:10]:
            if not time_ok():
                break
            if best_norm <= 1e-8:
                break
            try:
                x_result = damped_newton(x_init, max_iter=30)
                if x_result is not None:
                    update_best(x_result)
                    if best_norm <= 1e-8:
                        return best_x
            except:
                pass
    
    # Final: try Broyden from best_x with more iterations
    if best_norm > 1e-8 and best_norm < np.inf and time_ok():
        try:
            x_result = broyden_method(best_x, max_iter=50)
            update_best(x_result)
        except:
            pass
    
    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
