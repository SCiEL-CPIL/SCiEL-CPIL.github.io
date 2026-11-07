"""Root finder for 3 fixed 2-D systems: Secant-Step Hybrid Broyden with re-anchoring and polish."""
import numpy as np
import time
import scipy.optimize as opt


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Secant-Step Hybrid: initialize Jacobian via 2-point secant (2 calls),
       then alternate Broyden rank-1 updates with periodic full FD recomputation
       every k iterations to prevent quasi-Newton drift.
    2. If stuck, try restarts from scaled/random versions of x0.
    3. Final high-precision polish pass with tighter tolerance.
    
    Budget:
    - Time limit: ~118s internal (evaluator kills at 150s).
    - Call limit: < 500 calls per system.
    """
    start_time = time.time()
    time_limit = 118.0
    
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
    
    def compute_J_secant(x, Fx):
        """
        Two-point secant Jacobian estimate.
        Uses x and x + h*e (where e is a direction) to build J in 2 calls total
        (one for Fx already done, one for the perturbed point).
        Actually we need 2 additional calls for a 2x2 Jacobian via secant:
        J[:,0] ~ (F(x + h*e0) - F(x)) / h  -> 1 call
        J[:,1] ~ (F(x + h*e1) - F(x)) / h  -> 1 call
        That's 2 extra calls. Alternatively, use a single secant direction:
        Pick d = normalized Fx (or random if Fx ~ 0), evaluate F(x + h*d),
        then J ~ (F(x+h*d) - F(x)) * d^T / (d^T d)  -- rank-1, not full.
        
        Better: use 2 calls with two different perturbations to get a 2x2.
        We'll use forward differences in e0 and e1: 2 calls.
        """
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def compute_J_central(x, Fx):
        """
        Central-difference Jacobian: 4 calls (2 per column).
        More accurate for polish phase.
        """
        h = 1e-8
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_eval_F(x + e)
            Fx_m = safe_eval_F(x - e)
            if Fx_p is None or Fx_m is None:
                return None
            J[:, j] = (Fx_p - Fx_m) / (2 * h)
        return J

    def compute_J_forward(x, Fx):
        """
        Forward-difference Jacobian: 2 calls.
        Cheaper than central for polish when accuracy is sufficient.
        """
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_eval_F(x + e)
            if Fx_p is None:
                return None
            J[:, j] = (Fx_p - Fx) / h
        return J
    
    def broyden_hybrid(x_start, max_iter=40, reanchor_every=5):
        """
        Secant-Step Hybrid Broyden's method.
        - Initial J via forward differences (2 calls).
        - Rank-1 Broyden updates each iteration.
        - Every reanchor_every iterations, recompute J via forward differences (2 calls)
          to correct accumulated error.
        Returns (x, Fx, success).
        """
        x = np.array(x_start, dtype=float)
        
        # Initial evaluation
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian via forward differences (2 calls)
        J = compute_J_secant(x, Fx)
        if J is None:
            return x, Fx, False
        
        # Check invertibility
        if abs(np.linalg.det(J)) < 1e-12:
            # Try larger h
            h_backup = 1e-5
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h_backup
                Fx_e = safe_eval_F(x + e)
                if Fx_e is None:
                    return x, Fx, False
                J[:, j] = (Fx_e - Fx) / h_backup
            if abs(np.linalg.det(J)) < 1e-12:
                return x, Fx, False
        
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= 490:
                break
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Re-anchor before giving up
                if it < max_iter - 1:
                    J_new = compute_J_secant(x, Fx)
                    if J_new is not None and abs(np.linalg.det(J_new)) > 1e-12:
                        J = J_new
                        continue
                break
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
            
            x_new = x + dx
            
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                # Halve the step
                x_new = x + 0.5 * dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    x_new = x + 0.25 * dx
                    Fx_new = safe_eval_F(x_new)
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
                # Re-anchor: recompute J from scratch (2 calls)
                J_new = compute_J_secant(x_new, Fx_new)
                if J_new is not None and abs(np.linalg.det(J_new)) > 1e-12:
                    J = J_new
                elif s_dot_s > 1e-16:
                    # Fall back to Broyden update
                    J = J + np.outer(y - J @ s, s) / s_dot_s
            else:
                # Standard Broyden rank-1 update
                if s_dot_s > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    def polish(x_start):
        """
        High-precision polish: use forward-difference Newton for a few steps
        to squeeze below 1e-8 if close. Uses 2 calls per Jacobian instead of 4.
        """
        x = np.array(x_start, dtype=float)
        Fx = safe_eval_F(x)
        if Fx is None:
            return x
        norm = np.linalg.norm(Fx)
        if norm <= 1e-8:
            return x
        
        # Only polish if reasonably close
        if norm > 1e-4:
            return x
        
        for _ in range(6):
            if not time_ok() or call_count[0] >= 498:
                break
            
            # Use forward differences (2 calls) for cheaper polish
            J = compute_J_forward(x, Fx)
            if J is None:
                break
            if abs(np.linalg.det(J)) < 1e-14:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            step_norm = np.linalg.norm(dx)
            if step_norm > 0.5:
                dx = dx * (0.5 / step_norm)
            
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            if Fx_new_norm < norm:
                x = x_new
                Fx = Fx_new
                norm = Fx_new_norm
                if norm <= 1e-8:
                    return x
            else:
                # Try half step
                x_half = x + 0.5 * dx
                Fx_half = safe_eval_F(x_half)
                if Fx_half is not None:
                    Fx_half_norm = np.linalg.norm(Fx_half)
                    if Fx_half_norm < norm:
                        x = x_half
                        Fx = Fx_half
                        norm = Fx_half_norm
                        if norm <= 1e-8:
                            return x
                break
        
        return x
    
    # First attempt: Secant-Step Hybrid Broyden from x0
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden_hybrid(x0_arr, max_iter=40, reanchor_every=5)
    if success:
        return x
    
    # If not successful but close, try polish
    if Fx is not None and np.linalg.norm(Fx) <= 1e-4:
        x_polished = polish(x)
        Fx_check = safe_eval_F(x_polished)
        if Fx_check is not None and np.linalg.norm(Fx_check) <= 1e-8:
            return x_polished
    
    # If first attempt failed, try global optimization via L-BFGS-B on 0.5*||F(x)||^2
    def objective(x):
        Fx_val = safe_eval_F(x)
        if Fx_val is None:
            return 1e20
        return 0.5 * np.dot(Fx_val, Fx_val)
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    # Try L-BFGS-B from x0 and a few restart points
    start_points = [x0_arr.copy(), np.zeros(2), 0.5 * x0_arr, 2.0 * x0_arr]
    
    for sp in start_points:
        if not time_ok() or call_count[0] >= 460:
            break
        
        try:
            result = opt.minimize(objective, sp, method='L-BFGS-B', 
                                  options={'maxiter': 20, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.x is not None and result.x.shape == (2,):
                Fx_res = safe_eval_F(result.x)
                if Fx_res is not None:
                    norm_res = np.linalg.norm(Fx_res)
                    if norm_res < best_norm:
                        best_norm = norm_res
                        best_x = result.x.copy()
                    if norm_res <= 1e-8:
                        return result.x
        except Exception:
            pass
    
    # Also try from the best Broyden result
    if best_norm > 1e-8 and not time_ok() is False and call_count[0] < 480:
        try:
            result = opt.minimize(objective, best_x, method='L-BFGS-B',
                                  options={'maxiter': 15, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.x is not None and result.x.shape == (2,):
                Fx_res = safe_eval_F(result.x)
                if Fx_res is not None:
                    norm_res = np.linalg.norm(Fx_res)
                    if norm_res < best_norm:
                        best_norm = norm_res
                        best_x = result.x.copy()
                    if norm_res <= 1e-8:
                        return result.x
        except Exception:
            pass
    
    # Final polish on best candidate
    if best_norm <= 1e-4:
        x_polished = polish(best_x)
        Fx_check = safe_eval_F(x_polished)
        if Fx_check is not None and np.linalg.norm(Fx_check) <= 1e-8:
            return x_polished
        if Fx_check is not None:
            best_x = x_polished
    
    return best_x
