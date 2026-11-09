"""Root finder for 3 fixed two-dimensional systems using robust Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf
    
    try:
        # Initial evaluation
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return x
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        
        best_norm = norm_Fx
        
        # Initial Jacobian via Finite Differences
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_pert = F(x + e)
            Fx_pert = np.asarray(Fx_pert, dtype=float)
            if Fx_pert.shape != (2,) or not np.all(np.isfinite(Fx_pert)):
                # Fallback: use identity if FD fails? Or just try with current J?
                # Better to return best_x if we can't proceed
                return best_x
            J[:, j] = (Fx_pert - Fx) / h
        
        # Broyden's Method with Line Search
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for step: J * s = -Fx
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, add regularization or break
                J += 1e-6 * np.eye(2)
                try:
                    s = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
            
            # Line search
            alpha = 1.0
            x_new = x
            Fx_new = Fx
            found_step = False
            
            for _ in range(20):
                x_trial = x + alpha * s
                Fx_trial = F(x_trial)
                Fx_trial = np.asarray(Fx_trial, dtype=float)
                
                if Fx_trial.shape != (2,) or not np.all(np.isfinite(Fx_trial)):
                    alpha *= 0.5
                    continue
                
                norm_trial = np.linalg.norm(Fx_trial)
                
                # Armijo-like condition: decrease in norm
                if norm_trial < norm_Fx:
                    x_new = x_trial
                    Fx_new = Fx_trial
                    found_step = True
                    break
                
                alpha *= 0.5
                if alpha < 1e-10:
                    break
            
            if not found_step:
                # If no step reduces the norm, try a small random perturbation or break
                # Or just accept the current best
                break
            
            # Update Broyden Jacobian
            s_actual = x_new - x
            y = Fx_new - Fx
            denom = np.dot(s_actual, s_actual)
            if denom > 1e-14:
                J = J + np.outer(y - J @ s_actual, s_actual) / denom
            
            x = x_new
            Fx = Fx_new
            norm_Fx = np.linalg.norm(Fx)
            
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x.copy()
            
            if norm_Fx <= 1e-8:
                return x
        
        return best_x
        
    except Exception:
        return best_x if np.all(np.isfinite(best_x)) else x
# EVOLVE-BLOCK-END
