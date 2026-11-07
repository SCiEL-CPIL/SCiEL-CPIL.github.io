"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x
    
    best_x = x.copy()
    best_norm = norm_F
    
    # Parameters
    h = 1e-5  # Step size for finite differences
    max_iter = 60
    c_armijo = 1e-4  # Armijo constant
    max_alpha_halvings = 12
    
    # Jacobian reuse logic
    J = None
    use_reuse = False
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute Jacobian if not reusing
        if not use_reuse:
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = F(x + e)
                J[:, j] = (Fx_e - Fx) / h
            use_reuse = False # Reset flag after computing
        else:
            # We are reusing J, so we don't spend calls here
            pass

        # Solve J dx = -Fx
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Regularize if singular
            try:
                dx = -np.linalg.solve(J + 1e-6 * np.eye(2), Fx)
            except np.linalg.LinAlgError:
                # Fallback to gradient step
                dx = -Fx * 0.01
        
        # Limit step size
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm
        
        # Backtracking Line Search
        alpha = 1.0
        accepted = False
        Fx_new = None
        x_new = None
        
        for _ in range(max_alpha_halvings):
            x_candidate = x + alpha * dx
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            
            if norm_candidate <= 1e-8:
                return x_candidate
            
            if norm_candidate < best_norm:
                best_x = x_candidate.copy()
                best_norm = norm_candidate
                
            # Armijo condition
            if norm_candidate <= (1.0 - c_armijo * alpha) * norm_F:
                accepted = True
                x_new = x_candidate
                Fx_new = F_candidate
                break
            else:
                alpha *= 0.5
        
        if accepted:
            x = x_new
            Fx = Fx_new
            norm_F = np.linalg.norm(Fx)
            
            # Decide whether to reuse Jacobian
            # If full step was accepted and convergence is fast, reuse
            if alpha == 1.0 and norm_F < norm_F * 0.5: # This check is redundant, norm_F is updated
                # Actually, we need to compare with previous norm_F
                pass
            
            # Heuristic for reuse:
            # If we just did a full step (alpha=1.0) and the norm decreased significantly,
            # the Jacobian is likely still a good approximation.
            # Let's check the ratio.
            # We need the old norm_F. Let's store it before update.
            
            # Let's simplify: Reuse if alpha was 1.0 and we are not too far from root.
            # A safer heuristic: Reuse every 2nd iteration if alpha=1.0 was used.
            # Or just don't reuse to be safe, 2 calls is cheap.
            # However, to improve score, let's try to reuse if alpha=1.0.
            
            # Let's implement a simple counter.
            # If alpha == 1.0, set use_reuse = True for next iteration.
            # If alpha < 1.0, set use_reuse = False.
            if alpha == 1.0:
                use_reuse = True
            else:
                use_reuse = False
                
        else:
            # Line search failed
            # Try a small step in negative gradient direction
            # Gradient of 0.5 * ||F||^2 is J^T F
            grad = J.T @ Fx
            dx_grad = -grad
            grad_norm = np.linalg.norm(dx_grad)
            if grad_norm > 0:
                dx_grad = dx_grad / grad_norm * 0.1 # Small step
                
                x_candidate = x + dx_grad
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    use_reuse = False # Jacobian might be stale
                else:
                    # Stuck
                    break
            else:
                break
                
        if norm_F <= 1e-8:
            return x
            
    return best_x
# EVOLVE-BLOCK-END
