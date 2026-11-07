# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x of F: R^2 -> R^2 using continuation/homotopy method.
    
    Strategy:
    - Define homotopy F_t(x) = t*F(x) + (1-t)*(x - x0)
    - At t=0, root is exactly x0
    - Track root from t=0 to t=1 using Newton/Broyden with damping
    - Final polish with high-precision Newton
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf
    
    def check_deadline():
        return time.time() > deadline
    
    # Phase 1: Continuation from t=0 to t=1
    # Use adaptive t stepping with Newton at each step
    
    # Initial Jacobian estimate for Broyden
    def compute_jacobian_fd(x, Fx):
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        return J
    
    # Start with t=0, where F_0(x) = x - x0, root at x0
    t = 0.0
    x = x0.copy()
    
    # We'll track the root as t increases
    # Use a sequence of t values, adapting based on convergence
    
    t_values = [0.0]
    # Geometric progression toward 1.0
    # Start with larger steps, refine if needed
    target_ts = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.995, 0.999, 1.0]
    
    # Broyden Jacobian
    J = None
    prev_Ft = None
    prev_x = None
    
    for t_target in target_ts:
        if check_deadline():
            break
            
        # Solve for root of F_t at this t
        # Newton/Broyden iterations
        max_newton_iters = 30
        converged = False
        
        for it in range(max_newton_iters):
            if check_deadline():
                break
                
            Ft_x = t_target * F(x) + (1.0 - t_target) * (x - x0)
            norm_Ft = np.linalg.norm(Ft_x)
            
            if norm_Ft < best_norm:
                best_norm = norm_Ft
                best_x = x.copy()
            
            if norm_Ft <= 1e-14:
                converged = True
                break
            
            # Compute or update Jacobian
            if J is None:
                J = compute_jacobian_fd(x, Ft_x)
            else:
                # Broyden update if we have previous info
                if prev_Ft is not None and prev_x is not None:
                    s = x - prev_x
                    y = Ft_x - prev_Ft
                    # Check if update is valid
                    if np.linalg.norm(s) > 1e-14:
                        # Broyden's good method
                        rank1 = np.outer(y - J @ s, s) / (s @ s)
                        J = J + rank1
            
            # Newton step
            try:
                dx = np.linalg.solve(J, -Ft_x)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse or reset Jacobian
                J = compute_jacobian_fd(x, Ft_x)
                try:
                    dx = np.linalg.solve(J, -Ft_x)
                except np.linalg.LinAlgError:
                    # Last resort: gradient descent step
                    dx = -0.1 * Ft_x
            
            # Damped step with backtracking
            step = 1.0
            max_backtrack = 10
            for bt in range(max_backtrack):
                x_new = x + step * dx
                Ft_x_new = t_target * F(x_new) + (1.0 - t_target) * (x_new - x0)
                norm_new = np.linalg.norm(Ft_x_new)
                
                if norm_new < norm_Ft:
                    # Accept step
                    prev_Ft = Ft_x.copy()
                    prev_x = x.copy()
                    x = x_new
                    Ft_x = Ft_x_new
                    norm_Ft = norm_new
                    break
                else:
                    step *= 0.5
            
            if norm_Ft <= 1e-14:
                converged = True
                break
        
        if not converged and check_deadline():
            break
    
    # Phase 2: Final polish at t=1.0 with high precision
    # Use pure Newton on F(x) with tight tolerance
    if not check_deadline():
        # Reset for final polish
        J = None
        prev_Ft = None
        prev_x = None
        
        max_polish_iters = 50
        for it in range(max_polish_iters):
            if check_deadline():
                break
                
            Fx = F(x)
            norm_Fx = np.linalg.norm(Fx)
            
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x.copy()
            
            if norm_Fx <= 1e-12:
                break
            
            if J is None:
                J = compute_jacobian_fd(x, Fx)
            else:
                if prev_Ft is not None and prev_x is not None:
                    s = x - prev_x
                    y = Fx - prev_Ft
                    if np.linalg.norm(s) > 1e-14:
                        rank1 = np.outer(y - J @ s, s) / (s @ s)
                        J = J + rank1
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                J = compute_jacobian_fd(x, Fx)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    dx = -0.1 * Fx
            
            # Damped step
            step = 1.0
            for bt in range(10):
                x_new = x + step * dx
                Fx_new = F(x_new)
                norm_new = np.linalg.norm(Fx_new)
                
                if norm_new < norm_Fx:
                    prev_Ft = Fx.copy()
                    prev_x = x.copy()
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_new
                    break
                else:
                    step *= 0.5
    
    # Return the best result found
    # Check if best_x is valid
    if np.all(np.isfinite(best_x)) and best_x.shape == (2,):
        return best_x
    return x

# EVOLVE-BLOCK-END
