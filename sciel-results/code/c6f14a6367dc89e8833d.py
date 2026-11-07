import numpy as np
import time

def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2.
    Uses homotopy path-following from origin, then Broyden refinement.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin from 150s limit
    n = 2

    def norm(v):
        return float(np.linalg.norm(v))

    def safe_F(x):
        try:
            return np.array(F(x), dtype=float)
        except Exception:
            return np.array([1e10, 1e10])

    def compute_jacobian(x, Fx):
        """Central difference Jacobian using F(x) already computed."""
        h = 1e-5
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = safe_F(x + e)
            Fm = safe_F(x - e)
            J[:, j] = (Fp - Fm) / (2 * h)
        return J

    # Phase 1: Homotopy path-following from origin
    # We trace the solution of F_t(x) = (1-t)*F(x) + t*x = 0 from t=1 (x=0) to t=0
    # This gives us a path to the root
    
    best_x = np.array(x0, dtype=float)
    best_norm = norm(safe_F(best_x))
    
    if best_norm <= 1e-8:
        return best_x

    # Start from origin where F_1(0) = 0
    x_hom = np.zeros(2)
    t = 1.0  # Start at t=1 where solution is x=0
    
    # Adaptive RK4 step for homotopy
    dt_max = 0.1
    dt_min = 1e-6
    
    for hom_iter in range(200):
        if time.time() > deadline - 30:  # Reserve 30s for Broyden
            break
        if t <= 0:
            break
            
        # Try to take a step in t from current (x_hom, t) to (x_new, t_new)
        # We want to solve F_t(x) = (1-t)*F(x) + t*x = 0
        # At t=1: F_1(x) = x, so x=0 is solution
        # At t=0: F_0(x) = F(x), so we want root of F
        
        dt = dt_max
        step_success = False
        
        for try_dt in range(20):
            t_new = max(0.0, t - dt)
            
            # RK4 step for the homotopy ODE
            # dx/dt = -J^{-1} * dF_t/dt where dF_t/dt = -F(x) + x
            # But we use a simpler predictor-corrector
            
            # Predictor: use Newton-like step
            Fx = safe_F(x_hom)
            J = compute_jacobian(x_hom, Fx)
            
            try:
                # Solve for x_new such that F_{t_new}(x_new) ≈ 0
                # F_{t_new}(x) = (1-t_new)*F(x) + t_new*x
                # Use fixed-point iteration with damping
                x_curr = x_hom.copy()
                for _ in range(10):
                    F_curr = safe_F(x_curr)
                    F_t_curr = (1 - t_new) * F_curr + t_new * x_curr
                    try:
                        dx = np.linalg.solve(J, -F_t_curr)
                    except:
                        dx = -0.01 * F_t_curr
                    x_curr = x_curr + 0.5 * dx
                    if norm(dx) < 1e-10:
                        break
                
                x_new = x_curr
                F_new = safe_F(x_new)
                F_t_new = (1 - t_new) * F_new + t_new * x_new
                err = norm(F_t_new)
                
                if err < 1e-6:
                    x_hom = x_new
                    t = t_new
                    step_success = True
                    # Update best
                    F_check = safe_F(x_hom)
                    n_check = norm(F_check)
                    if n_check < best_norm:
                        best_x = x_hom.copy()
                        best_norm = n_check
                    if best_norm <= 1e-8:
                        return best_x
                    break
                else:
                    dt *= 0.5
                    if dt < dt_min:
                        break
                        
            except Exception:
                dt *= 0.5
                if dt < dt_min:
                    break
        
        if not step_success:
            # Try a different approach: direct Broyden from current point
            break
            
        # Increase step size if we succeeded easily
        if step_success and dt < dt_max:
            dt = min(dt_max, dt * 2)
    
    # Phase 2: Broyden refinement from best point found
    x_broyden = best_x.copy()
    Fx = safe_F(x_broyden)
    norm_Fx = norm(Fx)
    
    if norm_Fx <= 1e-8:
        return x_broyden
    
    J = compute_jacobian(x_broyden, Fx)
    
    max_iter = 30
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # Line search
        alpha = 1.0
        ls_success = False
        for _ in range(15):
            x_new = x_broyden + alpha * dx
            Fx_new = safe_F(x_new)
            norm_Fx_new = norm(Fx_new)
            
            if norm_Fx_new <= 1e-8:
                return x_new
            
            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new
            
            if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                s = x_new - x_broyden
                y = Fx_new - Fx
                s_dot_s = float(np.dot(s, s))
                if s_dot_s > 1e-30:
                    J += np.outer(y - J @ s, s) / s_dot_s
                x_broyden = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                ls_success = True
                break
            
            alpha *= 0.5
        
        if not ls_success:
            break
        
        if norm_Fx <= 1e-8:
            return x_broyden
    
    return best_x
