"""Root finder for 3 fixed 2-D systems: Broyden's method with safeguards and polish."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with adaptive Jacobian re-anchoring.
    2. Line search for robustness.
    3. Final polish with Newton's method when close to root.
    4. Adaptive restarts from promising points.
    
    Budget:
    - Time limit: ~145s internal (evaluator kills at 150s).
    - Call limit: < 500 calls per system.
    """
    start_time = time.time()
    time_limit = 145.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    call_count = [0]
    
    def safe_eval_F(x):
        """Evaluate F with error handling."""
        try:
            x_arr = np.array(x, dtype=float)
            if x_arr.shape != (2,) or not np.all(np.isfinite(x_arr)):
                call_count[0] += 1
                return None
            result = np.array(F(x_arr), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None
    
    def compute_jacobian_fd(x, Fx, h=1e-7):
        """Compute Jacobian using forward finite differences. Costs 2 calls."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def compute_jacobian_central(x, h=1e-6):
        """Compute Jacobian using central finite differences. Costs 4 calls."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_eval_F(x + e)
            Fx_minus = safe_eval_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                return None
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J
    
    def broyden_with_polish(x_start, max_iter=50, call_budget=450, polish_threshold=1e-4):
        """
        Broyden's method with line search and final Newton polish.
        Returns (x, Fx, success).
        """
        x = np.array(x_start, dtype=float).copy()
        
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian
        J = compute_jacobian_fd(x, Fx, h=1e-7)
        if J is None:
            J = compute_jacobian_fd(x, Fx, h=1e-5)
            if J is None:
                return x, Fx, False
        
        if abs(np.linalg.det(J)) < 1e-12:
            return x, Fx, False
        
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        # Track Jacobian quality
        jacobian_age = 0
        max_age_before_reanchor = 8
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_budget:
                break
            
            # Check if we should re-anchor Jacobian
            jacobian_age += 1
            should_reanchor = False
            if jacobian_age >= max_age_before_reanchor:
                should_reanchor = True
            else:
                # Check if Jacobian is degrading by comparing model prediction
                try:
                    dx_model = np.linalg.solve(J, -Fx)
                    # If the model step is very different from what we'd expect, re-anchor
                    if np.linalg.norm(dx_model) > 100:
                        should_reanchor = True
                except np.linalg.LinAlgError:
                    should_reanchor = True
            
            if should_reanchor:
                J_new = compute_jacobian_fd(x, Fx, h=1e-7)
                if J_new is None:
                    J_new = compute_jacobian_fd(x, Fx, h=1e-5)
                if J_new is not None and abs(np.linalg.det(J_new)) > 1e-12:
                    J = J_new
                    jacobian_age = 0
            
            # Solve for step
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                J_pert = J + 1e-8 * np.eye(2)
                try:
                    dx = np.linalg.solve(J_pert, -Fx)
                except np.linalg.LinAlgError:
                    break
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 20.0:
                dx = dx * (20.0 / step_norm)
            
            # Line search: try full step, then halves
            alpha = 1.0
            found_step = False
            for _ in range(8):
                x_new = x + alpha * dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    alpha *= 0.5
                    continue
                
                Fx_new_norm = np.linalg.norm(Fx_new)
                
                # Accept if improvement
                if Fx_new_norm < best_norm * 0.9 or Fx_new_norm < best_norm:
                    found_step = True
                    break
                
                alpha *= 0.5
            
            if not found_step:
                # Try a smaller step in the steepest descent direction
                grad_F = Fx.copy()  # For scalar, this is the gradient direction
                dx_sd = -grad_F / max(1.0, np.linalg.norm(grad_F))
                for alpha_try in [0.1, 0.01, 0.001]:
                    x_new = x + alpha_try * dx_sd
                    Fx_new = safe_eval_F(x_new)
                    if Fx_new is not None:
                        Fx_new_norm = np.linalg.norm(Fx_new)
                        if Fx_new_norm < best_norm:
                            found_step = True
                            dx = alpha_try * dx_sd
                            break
            
            if not found_step:
                break
            
            x_new = x + alpha * dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_Fx = Fx_new.copy()
                best_norm = Fx_new_norm
            
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            # Update Jacobian using Broyden's formula
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
            
            # If close to root, try Newton polish
            if best_norm < polish_threshold and call_count[0] < call_budget - 20:
                x_polish, Fx_polish, success_polish = newton_polish(best_x, best_Fx, max_iter=10, call_budget=call_budget)
                if success_polish:
                    return x_polish, Fx_polish, True
                if Fx_polish is not None:
                    norm_polish = np.linalg.norm(Fx_polish)
                    if norm_polish < best_norm:
                        best_x = x_polish.copy()
                        best_Fx = Fx_polish.copy()
                        best_norm = norm_polish
                        x = x_polish.copy()
                        Fx = Fx_polish.copy()
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    def newton_polish(x_start, Fx_start, max_iter=10, call_budget=450):
        """
        Newton's method with central difference Jacobian for final polish.
        Returns (x, Fx, success).
        """
        x = np.array(x_start, dtype=float).copy()
        Fx = np.array(Fx_start, dtype=float).copy()
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_budget:
                break
            
            J = compute_jacobian_central(x, h=1e-6)
            if J is None:
                J = compute_jacobian_fd(x, Fx, h=1e-6)
                if J is None:
                    break
            
            if abs(np.linalg.det(J)) < 1e-12:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            x = x_new
            Fx = Fx_new
        
        return x, Fx, np.linalg.norm(Fx) <= 1e-8 if Fx is not None else False
    
    # First attempt: Broyden from x0
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden_with_polish(x0_arr, max_iter=50, call_budget=450)
    if success:
        return x
    
    # Adaptive restarts
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    # Generate diverse restart points
    restarts = []
    n0_norm = np.linalg.norm(x0_arr)
    
    if n0_norm < 1e-10:
        # x0 is near origin, try various directions
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            restarts.append(np.array([np.cos(angle), np.sin(angle)]))
    else:
        # Scaled versions
        for scale in [0.5, 2.0, 0.1, 5.0, -1.0, -0.5, 0.2, 10.0]:
            restarts.append(scale * x0_arr)
        
        # Perpendicular directions
        perp1 = np.array([-x0_arr[1], x0_arr[0]])
        perp2 = np.array([x0_arr[1], -x0_arr[0]])
        for p in [perp1, perp2]:
            p_norm = np.linalg.norm(p)
            if p_norm > 1e-10:
                p = p / p_norm * n0_norm
                restarts.append(p)
                restarts.append(x0_arr + p)
                restarts.append(x0_arr - p)
    
    # Points in direction of -F(x0) (steepest descent from x0)
    Fx0 = safe_eval_F(x0_arr)
    if Fx0 is not None:
        Fx0_norm = np.linalg.norm(Fx0)
        if Fx0_norm > 1e-10:
            direction = -Fx0 / Fx0_norm
            for step in [0.1, 0.5, 1.0, 2.0]:
                restarts.append(x0_arr + step * n0_norm * direction)
    
    # Random points
    rng = np.random.RandomState(42)
    for _ in range(8):
        restarts.append(rng.randn(2) * max(1.0, n0_norm))
    
    # Origin
    restarts.append(np.zeros(2))
    
    # Try restarts, prioritizing those that seem promising
    for r in restarts:
        if not time_ok() or call_count[0] >= 470:
            break
        
        x_r, Fx_r, success_r = broyden_with_polish(r, max_iter=30, call_budget=470)
        if success_r:
            return x_r
        
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = x_r.copy()
    
    return best_x
