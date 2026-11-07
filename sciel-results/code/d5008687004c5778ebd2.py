"""Root finder for 3 fixed 2-D systems: Secant-Step Hybrid with re-anchoring."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with periodic full Jacobian recomputation (re-anchoring).
    2. Secant-step hybrid: alternate between rank-1 updates and 3-call FD Jacobian.
    3. Damped steps with line search for robustness.
    4. Multiple restarts if initial attempt fails.
    
    Budget:
    - Time limit: ~115s internal (evaluator kills at 150s).
    - Call limit: < 500 calls per system.
    """
    start_time = time.time()
    time_limit = 115.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    # Safe F evaluation that counts calls and handles errors
    call_count = [0]
    
    def safe_eval_F(x):
        """Evaluate F with error handling."""
        try:
            result = np.array(F(x), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None
    
    def compute_J_fd(x, Fx, h=1e-6):
        """Compute Jacobian via central finite differences (3 calls: x, x+e1, x+e2)."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            # Central difference: (F(x+e) - F(x-e)) / (2h)
            # But we already have Fx = F(x), so use forward: (F(x+e) - Fx) / h
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def compute_J_central(x, h=1e-5):
        """Compute Jacobian via central finite differences (4 calls)."""
        Fx0 = safe_eval_F(x)
        if Fx0 is None:
            return None, None
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_eval_F(x + e)
            Fx_minus = safe_eval_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                return None, None
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J, Fx0
    
    def broyden_hybrid(x_start, max_iter=30, reanchor_every=5):
        """
        Broyden's method with periodic re-anchoring via full FD Jacobian.
        Returns (x, Fx, success) where success is True if ||Fx|| <= 1e-8.
        """
        x = np.array(x_start, dtype=float)
        
        # Initial evaluation
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian estimate using forward differences (2 calls)
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return x, Fx, False
            J[:, j] = (Fx_e - Fx) / h
        
        # Check if J is invertible
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-12:
            # Try a different perturbation
            h = 1e-4
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = safe_eval_F(x + e)
                if Fx_e is None:
                    return x, Fx, False
                J[:, j] = (Fx_e - Fx) / h
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-12:
                return x, Fx, False
        
        # Broyden's iterations with re-anchoring
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= 470:
                break
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Try to recover by recomputing J
                J, Fx_new_J = compute_J_central(x)
                if J is None:
                    break
                if np.linalg.norm(Fx_new_J) <= 1e-8:
                    return x, Fx_new_J, True
                try:
                    dx = np.linalg.solve(J, -Fx_new_J)
                except np.linalg.LinAlgError:
                    break
                Fx = Fx_new_J
                if np.linalg.norm(Fx) < best_norm:
                    best_x = x.copy()
                    best_Fx = Fx.copy()
                    best_norm = np.linalg.norm(Fx)
                continue
            
            # Damping: if step is too large, reduce it
            step_norm = np.linalg.norm(dx)
            if step_norm > 50.0:
                dx = dx * (50.0 / step_norm)
            
            # Line search: try full step, then halve
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                # Try smaller steps
                for alpha in [0.5, 0.25, 0.1]:
                    x_new = x + alpha * dx
                    Fx_new = safe_eval_F(x_new)
                    if Fx_new is not None:
                        break
                if Fx_new is None:
                    break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            # Track best
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_Fx = Fx_new.copy()
                best_norm = Fx_new_norm
            
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            # Update Jacobian
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            
            if (it + 1) % reanchor_every == 0:
                # Re-anchor: recompute full Jacobian
                J_new, Fx_reanchor = compute_J_central(x_new)
                if J_new is not None:
                    J = J_new
                    Fx = Fx_reanchor
                    if np.linalg.norm(Fx) < best_norm:
                        best_x = x_new.copy()
                        best_Fx = Fx.copy()
                        best_norm = np.linalg.norm(Fx)
                    x = x_new
                    if np.linalg.norm(Fx) <= 1e-8:
                        return x, Fx, True
                    continue
            else:
                # Standard Broyden rank-1 update
                if s_dot_s > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    # First attempt: Broyden hybrid from x0
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden_hybrid(x0_arr, max_iter=30, reanchor_every=5)
    if success:
        return x
    
    # If first attempt failed, try restarts
    restarts = []
    
    # Scaled versions of x0
    for scale in [0.5, 2.0, 0.1, 5.0, -1.0, -0.5]:
        restarts.append(scale * x0_arr)
    
    # Random perturbations
    rng = np.random.RandomState(42)
    for _ in range(8):
        restarts.append(x0_arr + rng.randn(2) * max(np.linalg.norm(x0_arr), 1.0) * 0.5)
    
    # Origin
    restarts.append(np.zeros(2))
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    for r in restarts:
        if not time_ok() or call_count[0] >= 440:
            break
        
        x_r, Fx_r, success_r = broyden_hybrid(r, max_iter=25, reanchor_every=5)
        if success_r:
            return x_r
        
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = x_r.copy()
    
    # Final check on best_x
    if Fx is not None and np.linalg.norm(Fx) <= 1e-8:
        return best_x
    
    return best_x
