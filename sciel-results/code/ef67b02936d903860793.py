# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method with
    adaptive line search and deadline guard.
    """
    x = np.array(x0, dtype=float).reshape(2)
    
    # Deadline guard: 148s safety margin before 150s kill
    start_time = time.time()
    deadline = start_time + 148.0
    
    def time_up():
        return time.time() > deadline
    
    def is_valid_point(x):
        return np.all(np.isfinite(x)) and x.shape == (2,)
    
    def norm(v):
        return float(np.linalg.norm(v))
    
    # Initial evaluation
    if not is_valid_point(x):
        x = np.array([0.0, 0.0])
    
    Fx = F(x)
    Fx = np.asarray(Fx, dtype=float).reshape(2)
    
    if not np.all(np.isfinite(Fx)):
        # Try a different starting point
        x = np.array([1.0, 1.0])
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float).reshape(2)
    
    if norm(Fx) <= 1e-8:
        return x
    
    # Initialize Jacobian approximation via finite differences
    eps = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        Fx_plus = F(x + e)
        Fx_plus = np.asarray(Fx_plus, dtype=float).reshape(2)
        J[:, j] = (Fx_plus - Fx) / eps
    
    best_x = x.copy()
    best_Fx = Fx.copy()
    best_norm = norm(Fx)
    
    max_iter = 100
    
    for iteration in range(max_iter):
        if time_up():
            break
        
        # Check convergence
        if best_norm <= 1e-8:
            return best_x
        
        # Solve for Newton step
        try:
            step = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use steepest descent direction
            step = -Fx
            step_norm = norm(step)
            if step_norm < 1e-15:
                break
            step = step / step_norm
        
        # Adaptive line search
        alpha = 1.0
        c1 = 1e-4  # Armijo condition
        max_line_search = 20
        
        for ls_iter in range(max_line_search):
            if time_up():
                break
            
            x_trial = x + alpha * step
            if not is_valid_point(x_trial):
                alpha *= 0.5
                continue
            
            Fx_trial = F(x_trial)
            Fx_trial = np.asarray(Fx_trial, dtype=float).reshape(2)
            
            if not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue
            
            Fx_trial_norm = norm(Fx_trial)
            
            # Track best
            if Fx_trial_norm < best_norm:
                best_norm = Fx_trial_norm
                best_x = x_trial.copy()
                best_Fx = Fx_trial.copy()
            
            if Fx_trial_norm <= 1e-8:
                return x_trial
            
            # Armijo condition: sufficient decrease
            if Fx_trial_norm <= norm(Fx) + c1 * alpha * np.dot(step, Fx):
                # Accept step
                s = x_trial - x
                y = Fx_trial - Fx
                x = x_trial
                Fx = Fx_trial
                Fx_norm = Fx_trial_norm
                
                # Broyden update
                if norm(s) > 1e-15 and norm(y) > 1e-15:
                    denom = np.dot(s, y)
                    if abs(denom) > 1e-15:
                        J = J + np.outer(y - J @ s, s) / denom
                break
            else:
                alpha *= 0.5
        
        # If we didn't accept a step, try a smaller step or restart
        if alpha < 1e-10:
            # Try a random perturbation
            x_trial = x + 0.1 * np.random.randn(2)
            if not is_valid_point(x_trial):
                x_trial = x + 0.1 * np.array([1.0, 0.0])
            Fx_trial = F(x_trial)
            Fx_trial = np.asarray(Fx_trial, dtype=float).reshape(2)
            if np.all(np.isfinite(Fx_trial)):
                Fx_trial_norm = norm(Fx_trial)
                if Fx_trial_norm < best_norm:
                    x = x_trial
                    Fx = Fx_trial
                    best_norm = Fx_trial_norm
                    best_x = x_trial.copy()
                    best_Fx = Fx_trial.copy()
                    # Recompute Jacobian
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = eps
                        Fx_plus = F(x + e)
                        Fx_plus = np.asarray(Fx_plus, dtype=float).reshape(2)
                        J[:, j] = (Fx_plus - Fx) / eps
    
    # Return best found
    if best_norm <= 1e-8:
        return best_x
    return best_x
# EVOLVE-BLOCK-END
