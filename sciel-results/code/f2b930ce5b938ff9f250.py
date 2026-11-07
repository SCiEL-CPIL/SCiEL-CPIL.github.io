# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Broyden's method with
    adaptive line search, Jacobian refresh, and fallback strategies.
    
    F: callable, takes array of shape (2,), returns array of shape (2,)
    x0: initial guess, array of shape (2,)
    Returns: x such that ||F(x)|| <= 1e-8
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    x = np.array(x0, dtype=float).flatten()
    if x.shape != (2,):
        x = x.reshape(2)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    # Check initial point
    Fx = F(x)
    Fx = np.asarray(Fx, dtype=float).flatten()
    if not np.all(np.isfinite(Fx)):
        # Try a small perturbation
        x = x + np.array([1e-3, -1e-3])
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float).flatten()
        if not np.all(np.isfinite(Fx)):
            return x
    
    norm_F = np.linalg.norm(Fx)
    if norm_F <= 1e-8:
        return x
    
    best_x = x.copy()
    best_norm = norm_F
    
    # Estimate initial Jacobian via finite differences
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            h = 1e-7 * max(1.0, abs(x[j]))
            e[j] = h
            Fx_plus = F(x + e)
            Fx_plus = np.asarray(Fx_plus, dtype=float).flatten()
            if not np.all(np.isfinite(Fx_plus)):
                Fx_plus = Fx
            J[:, j] = (Fx_plus - Fx) / h
        return J
    
    J = compute_jacobian(x, Fx)
    
    # Check if J is singular
    try:
        det = np.linalg.det(J)
        if abs(det) < 1e-14:
            J += 1e-12 * np.eye(2)
    except np.linalg.LinAlgError:
        J += 1e-12 * np.eye(2)
    
    # Main iteration loop with Broyden's method
    for iteration in range(150):
        if time_up():
            break
        
        if norm_F <= 1e-8:
            return x
        
        # Try damped Newton step
        try:
            cond = np.linalg.cond(J)
            if cond > 1e12:
                J_reg = J + 1e-12 * np.eye(2)
                direction = np.linalg.solve(J_reg, -Fx)
            else:
                direction = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If Newton fails, try a steepest descent step
            direction = -Fx
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1.0:
                direction = direction / dir_norm
        
        # Line search with backtracking
        step_size = 1.0
        max_backtrack = 25
        found_better = False
        
        for backtrack in range(max_backtrack):
            if time_up():
                break
            
            new_x = x + step_size * direction
            
            if not np.all(np.isfinite(new_x)):
                step_size *= 0.5
                continue
            
            new_Fx = F(new_x)
            new_Fx = np.asarray(new_Fx, dtype=float).flatten()
            if not np.all(np.isfinite(new_Fx)):
                step_size *= 0.5
                continue
            
            new_norm = np.linalg.norm(new_Fx)
            
            if new_norm <= 1e-8:
                return new_x
            
            # Armijo condition for line search
            if new_norm < norm_F:
                # Update Broyden's formula
                s = step_size * direction
                y = new_Fx - Fx
                # Rank-1 update: J_new = J + (y - J*s) @ s.T / (s.T @ s)
                s_dot_s = np.dot(s, s)
                if s_dot_s > 1e-20:
                    J = J + (y - J @ s) @ s.T / s_dot_s
                
                x = new_x
                Fx = new_Fx
                norm_F = new_norm
                found_better = True
                
                if norm_F <= best_norm:
                    best_x = x.copy()
                    best_norm = norm_F
                break
            else:
                step_size *= 0.5
        
        if found_better:
            continue
        
        # No progress made with current Jacobian
        # Try restarting with fresh finite difference Jacobian
        J_new = compute_jacobian(x, Fx)
        
        # Check if new J is better conditioned
        try:
            cond_old = np.linalg.cond(J)
        except np.linalg.LinAlgError:
            cond_old = np.inf
        try:
            cond_new = np.linalg.cond(J_new)
        except np.linalg.LinAlgError:
            cond_new = np.inf
        
        if cond_new < cond_old or np.isinf(cond_old):
            J = J_new
        else:
            # Add regularization
            J = J + 1e-10 * np.eye(2)
        
        # If still no progress, try a small step in negative gradient direction
        grad = J.T @ Fx
        if np.linalg.norm(grad) > 1e-15:
            step_size = 0.1 / np.linalg.norm(grad)
            x_new = x - step_size * grad
            Fx_new = F(x_new)
            Fx_new = np.asarray(Fx_new, dtype=float).flatten()
            
            if np.all(np.isfinite(Fx_new)):
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < norm_F:
                    x = x_new
                    Fx = Fx_new
                    norm_F = norm_new
                    if norm_F <= best_norm:
                        best_x = x.copy()
                        best_norm = norm_F
        
        # Check if x is getting too large
        if np.linalg.norm(x) > 1e10:
            break
    
    # Return best found
    if best_norm <= 1e-8:
        return best_x
    if best_norm < norm_F:
        return best_x
    return x
# EVOLVE-BLOCK-END
