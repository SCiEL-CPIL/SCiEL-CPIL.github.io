"""Root finder for 3 fixed 2D systems using damped Newton with adaptive step and multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Deadline guard: 150s limit, use 140s for safety
    start_time = time.time()
    time_limit = 140.0
    
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
        
    tol = 1e-8
    max_iter = 50
    max_restarts = 8
    fd_step = 1e-7
    
    best_x = x.copy()
    best_Fx = None
    best_norm = np.inf
    
    # Multi-start points: original x0 plus diverse perturbations
    start_points = [x0.copy()]
    for scale in [0.1, 0.3, 0.7, 1.5, 3.0, 5.0]:
        for angle in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
            offset = scale * np.array([np.cos(angle), np.sin(angle)])
            start_points.append(x0 + offset)
    
    for restart in range(max_restarts):
        if restart < len(start_points):
            x = start_points[restart].copy()
        else:
            # If we've used all predefined starts, try random perturbations
            base = best_x if best_norm < np.inf else x0
            x = base + 1.0 * (np.random.rand(2) - 0.5)
        
        if not np.all(np.isfinite(x)):
            x = base if 'base' in locals() else x0
        
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                continue
            
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x.copy()
                best_Fx = Fx.copy()
                
            if norm_Fx <= tol:
                return x
            
            # Compute initial Jacobian using finite differences
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = fd_step
                Fx_pert = F(x + e)
                Fx_pert = np.asarray(Fx_pert, dtype=float).reshape(2)
                J[:, j] = (Fx_pert - Fx) / fd_step
            
            # Damped Newton Loop with adaptive damping
            alpha = 1.0
            for i in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                
                # Solve for Newton step direction
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    # If singular, use pseudo-inverse
                    try:
                        dx = np.linalg.pinv(J) @ (-Fx)
                    except Exception:
                        break
                
                # Adaptive damping: start with current alpha, reduce if needed
                accepted = False
                for _ in range(15): # Max 15 damping steps
                    x_new = x + alpha * dx
                    Fx_new = F(x_new)
                    Fx_new = np.asarray(Fx_new, dtype=float).reshape(2)
                    
                    if not np.all(np.isfinite(Fx_new)):
                        alpha *= 0.5
                        continue
                    
                    norm_Fx_new = np.linalg.norm(Fx_new)
                    
                    # Accept if we make progress
                    if norm_Fx_new < norm_Fx * 0.99:  # Require 1% improvement
                        # Update Jacobian using rank-1 update (Broyden-like)
                        delta_x = alpha * dx
                        delta_F = Fx_new - Fx
                        
                        denom = np.dot(delta_x, delta_x)
                        if denom > 1e-14:
                            update_term = (delta_F - J @ delta_x) / denom
                            J += np.outer(update_term, delta_x)
                        
                        # Accept step
                        x = x_new
                        Fx = Fx_new
                        norm_Fx = norm_Fx_new
                        accepted = True
                        
                        if norm_Fx < best_norm:
                            best_norm = norm_Fx
                            best_x = x.copy()
                            best_Fx = Fx.copy()
                            
                        if norm_Fx <= tol:
                            return x
                        
                        # Adapt alpha: increase if we had to damp, decrease if full step worked
                        if alpha < 0.9:
                            alpha = min(1.0, alpha * 1.5)
                        break
                    else:
                        alpha *= 0.5
                
                if not accepted:
                    # If no step was accepted, try a small random perturbation
                    x = x + 0.01 * (np.random.rand(2) - 0.5)
                    Fx = F(x)
                    Fx = np.asarray(Fx, dtype=float).reshape(2)
                    if not np.all(np.isfinite(Fx)):
                        break
                    norm_Fx = np.linalg.norm(Fx)
                    
                    # Recompute Jacobian
                    J = np.zeros((2, 2))
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = fd_step
                        Fx_pert = F(x + e)
                        Fx_pert = np.asarray(Fx_pert, dtype=float).reshape(2)
                        J[:, j] = (Fx_pert - Fx) / fd_step
                    alpha = 0.5
                    
        except Exception:
            continue
            
    return best_x
# EVOLVE-BLOCK-END
