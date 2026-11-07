import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds a root x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with finite difference initialization,
    polish phase, and robust line search.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float).reshape(2)
    Fx = np.array(F(x), dtype=float)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = norm_F
    total_calls = 1
    
    # Constants
    max_iter = 100
    fd_h = 1e-5
    polish_tol = 1e-4
    
    def update_best(x_cand, norm_cand):
        nonlocal best_x, best_norm
        if norm_cand < best_norm:
            best_x = x_cand.copy()
            best_norm = norm_cand

    # Compute initial Jacobian using forward finite differences
    def compute_fd_jacobian(x_curr, F_curr):
        J_fd = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = fd_h
            Fx_e = np.array(F(x_curr + e), dtype=float)
            J_fd[:, j] = (Fx_e - F_curr) / fd_h
        return J_fd

    J = compute_fd_jacobian(x, Fx)
    total_calls += 2
    
    # Broyden's Method Loop
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check for convergence
        if norm_F <= 1e-8:
            return x
            
        # Check if we're in polish zone - use exact Newton with reused J
        if norm_F < polish_tol:
            # Use the current Broyden J as an approximation for Newton steps
            # This avoids FD calls in the polish phase
            try:
                s = -np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                # Fall back to FD if J is singular
                J = compute_fd_jacobian(x, Fx)
                total_calls += 2
                try:
                    s = -np.linalg.solve(J, Fx)
                except np.linalg.LinAlgError:
                    break
            
            # Try full step first
            x_trial = x + s
            F_trial = np.array(F(x_trial), dtype=float)
            norm_trial = np.linalg.norm(F_trial)
            total_calls += 1
            
            if not np.all(np.isfinite(F_trial)):
                # Non-finite, backtrack
                alpha = 0.5
                for _ls in range(5):
                    x_trial = x + alpha * s
                    F_trial = np.array(F(x_trial), dtype=float)
                    norm_trial = np.linalg.norm(F_trial)
                    total_calls += 1
                    if not np.all(np.isfinite(F_trial)):
                        alpha *= 0.5
                        continue
                    if norm_trial <= 1e-8:
                        return x_trial
                    if norm_trial < norm_F:
                        s_actual = x_trial - x
                        y = F_trial - Fx
                        if s_actual @ s_actual > 1e-15:
                            J = J + np.outer(y - J @ s_actual, s_actual) / (s_actual @ s_actual)
                        x = x_trial
                        Fx = F_trial
                        norm_F = norm_trial
                        update_best(x, norm_F)
                        break
                    else:
                        alpha *= 0.5
                continue
            
            if norm_trial <= 1e-8:
                return x_trial
            
            if norm_trial < norm_F:
                # Update J with Broyden
                s_actual = x_trial - x
                y = F_trial - Fx
                if s_actual @ s_actual > 1e-15:
                    J = J + np.outer(y - J @ s_actual, s_actual) / (s_actual @ s_actual)
                
                x = x_trial
                Fx = F_trial
                norm_F = norm_trial
                update_best(x, norm_F)
            else:
                # Backtrack
                alpha = 0.5
                for _ls in range(5):
                    x_trial = x + alpha * s
                    F_trial = np.array(F(x_trial), dtype=float)
                    norm_trial = np.linalg.norm(F_trial)
                    total_calls += 1
                    
                    if not np.all(np.isfinite(F_trial)):
                        alpha *= 0.5
                        continue
                        
                    if norm_trial <= 1e-8:
                        return x_trial
                    
                    if norm_trial < norm_F:
                        s_actual = x_trial - x
                        y = F_trial - Fx
                        if s_actual @ s_actual > 1e-15:
                            J = J + np.outer(y - J @ s_actual, s_actual) / (s_actual @ s_actual)
                        
                        x = x_trial
                        Fx = F_trial
                        norm_F = norm_trial
                        update_best(x, norm_F)
                        break
                    else:
                        alpha *= 0.5
                else:
                    # If all backtracks failed, recompute J and continue
                    J = compute_fd_jacobian(x, Fx)
                    total_calls += 2
            continue
        
        # Regular Broyden step
        try:
            # Check condition number; if ill-conditioned, recompute J
            cond = np.linalg.cond(J)
            if cond > 1e12:
                J = compute_fd_jacobian(x, Fx)
                total_calls += 2
                continue
            
            s = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Singular matrix, recompute Jacobian
            J = compute_fd_jacobian(x, Fx)
            total_calls += 2
            continue
        
        # Limit step size to avoid huge jumps
        step_norm = np.linalg.norm(s)
        if step_norm > 10.0:
            s = s / step_norm * 10.0
        
        # Line Search: Try to accept the step
        # We want a reduction in the norm.
        accepted = False
        alpha = 1.0
        max_ls = 5
        
        # If we are close, we might just take the full step if it reduces norm
        # Otherwise, backtrack
        for ls_iter in range(max_ls):
            x_trial = x + alpha * s
            F_trial = np.array(F(x_trial), dtype=float)
            norm_trial = np.linalg.norm(F_trial)
            total_calls += 1
            
            if not np.all(np.isfinite(F_trial)):
                alpha *= 0.5
                continue
                
            if norm_trial <= 1e-8:
                return x_trial
            
            update_best(x_trial, norm_trial)
            
            # Accept if we have significant progress
            # For Broyden, even a small reduction is good.
            if norm_trial < norm_F:
                accepted = True
                # Broyden Update
                # s_actual = x_trial - x
                # y = F_trial - Fx
                s_actual = x_trial - x
                y = F_trial - Fx
                
                # Broyden-D (Good for 2D)
                # J_new = J + (y - J*s) * s^T / (s^T * s)
                if s_actual @ s_actual > 1e-15:
                    J = J + np.outer(y - J @ s_actual, s_actual) / (s_actual @ s_actual)
                
                x = x_trial
                Fx = F_trial
                norm_F = norm_trial
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            # If line search failed, the Broyden direction might be bad or J is stale.
            # Strategy: Recompute Jacobian with FD and try again.
            J = compute_fd_jacobian(x, Fx)
            total_calls += 2
            
            # If we still can't make progress, try a small step in negative gradient direction
            # Gradient of 0.5*||F||^2 is J^T F
            # But since we don't have exact J, let's just try moving opposite to F (assuming J~I locally)
            # This is a last resort to escape local minima of ||F||
            s_fallback = -Fx * 0.1
            x_trial = x + s_fallback
            F_trial = np.array(F(x_trial), dtype=float)
            norm_trial = np.linalg.norm(F_trial)
            total_calls += 1
            
            if not np.all(np.isfinite(F_trial)):
                break
                
            if norm_trial < norm_F:
                x = x_trial
                Fx = F_trial
                norm_F = norm_trial
                update_best(x, norm_F)
                # Recompute J again to restart Broyden
                J = compute_fd_jacobian(x, Fx)
                total_calls += 2
            else:
                # Stuck? Break to avoid wasting calls
                break
                
    # Final check
    if norm_F <= 1e-8:
        return x
    return best_x
