"""Root finder for 3 fixed 2-D systems: Broyden with trust-region safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method (rank-1 update) for fast convergence near the root.
    2. Trust-region damping: if step is too large or doesn't reduce residual,
       damp the step.
    3. Time guard: stop before 115s and return best result found.
    4. Fallback: if Broyden fails, try damped Newton with finite-difference Jacobian.
    
    Budget:
    - Time limit: 115s internal deadline (120s hard limit).
    - Call limit: < 500 calls per system.
    """
    t_start = time.time()
    deadline = t_start + 115.0
    x = np.array(x0, dtype=float)
    Fx = F(x)
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    def time_ok():
        return time.time() < deadline
    
    # Broyden's method with trust-region damping
    # Initial Jacobian: identity (or finite difference if we have budget)
    J = np.eye(2)
    
    # Try to get a better initial Jacobian with finite differences if close to root
    if best_norm < 1.0:
        # Use finite difference Jacobian for initial guess
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        # Update best if F changed
        # (Fx is still valid)
    
    max_iter = 40
    for it in range(max_iter):
        if not time_ok():
            break
        if best_norm <= 1e-8:
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use damped Newton with finite-difference Jacobian
            break
        
        # Trust-region damping
        step_norm = np.linalg.norm(dx)
        # Limit step size to avoid overshooting
        max_step = 1.0
        if step_norm > max_step:
            dx = dx * (max_step / step_norm)
            step_norm = max_step
        
        x_new = x + dx
        Fx_new = F(x_new)
        norm_new = np.linalg.norm(Fx_new)
        
        # Update best
        if norm_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_new
        
        # Check convergence
        if norm_new <= 1e-8:
            x = x_new
            Fx = Fx_new
            break
        
        # Broyden rank-1 update: J = J + (y - J*dx) * dx^T / (dx^T * dx)
        y = Fx_new - Fx
        denom = dx @ dx
        if denom > 1e-14:
            J = J + np.outer(y - J @ dx, dx) / denom
        
        # Accept step if it reduces the residual
        if norm_new < best_norm:
            x = x_new
            Fx = Fx_new
        else:
            # Reject step, try smaller step
            # Halve the step
            dx = dx * 0.5
            x_new = x + dx
            Fx_new = F(x_new)
            norm_new = np.linalg.norm(Fx_new)
            
            if norm_new < best_norm:
                x = x_new
                Fx = Fx_new
                best_x = x.copy()
                best_norm = norm_new
                if norm_new <= 1e-8:
                    break
                # Update Broyden
                y = Fx_new - Fx
                denom = dx @ dx
                if denom > 1e-14:
                    J = J + np.outer(y - J @ dx, dx) / denom
            else:
                # If even half step doesn't help, try a different direction
                # Use steepest descent direction: -Fx
                dx = -Fx * 0.1
                x_new = x + dx
                Fx_new = F(x_new)
                norm_new = np.linalg.norm(Fx_new)
                
                if norm_new < best_norm:
                    x = x_new
                    Fx = Fx_new
                    best_x = x.copy()
                    best_norm = norm_new
                    if norm_new <= 1e-8:
                        break
                    # Update Broyden
                    y = Fx_new - Fx
                    denom = dx @ dx
                    if denom > 1e-14:
                        J = J + np.outer(y - J @ dx, dx) / denom
                else:
                    # Stagnation: break and try fallback
                    break
    
    if best_norm <= 1e-8:
        return best_x
    
    # Fallback: Damped Newton with finite-difference Jacobian
    x = best_x.copy()
    Fx = F(x)
    best_norm = np.linalg.norm(Fx)
    
    for it in range(20):
        if not time_ok():
            break
        if best_norm <= 1e-8:
            break
        
        # Finite-difference Jacobian
        h = 1e-6
        J_fd = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J_fd[:, j] = (F(x + e) - Fx) / h
        
        try:
            dx = np.linalg.solve(J_fd, -Fx)
        except np.linalg.LinAlgError:
            break
        
        # Damped Newton: line search
        alpha = 1.0
        for _ in range(10):
            x_new = x + alpha * dx
            Fx_new = F(x_new)
            norm_new = np.linalg.norm(Fx_new)
            
            if norm_new < best_norm:
                x = x_new
                Fx = Fx_new
                best_x = x.copy()
                best_norm = norm_new
                if norm_new <= 1e-8:
                    break
                break
            else:
                alpha *= 0.5
        
        if best_norm <= 1e-8:
            break
    
    return best_x
# EVOLVE-BLOCK-END
