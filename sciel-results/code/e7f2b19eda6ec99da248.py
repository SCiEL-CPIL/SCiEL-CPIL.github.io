"""Root finder for 3 fixed 2D systems using Broyden's method with robust restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - time.time()
        
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf
    
    # Initial Jacobian estimation
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        return J

    # Broyden's Method
    Fx = F(x)
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx < 1e-8:
        return x
        
    J = compute_jacobian(x, Fx)
    
    max_iters = 100
    for i in range(max_iters):
        if time_left() < 5.0:
            break
            
        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, perturb or restart
            # Simple fallback: use gradient descent step on ||F||^2
            # grad = J.T Fx
            # s = -grad
            try:
                s = -J.T @ Fx
            except:
                break
                
        # Line search to ensure decrease in ||F||
        alpha = 1.0
        x_new = x + alpha * s
        Fx_new = F(x_new)
        norm_Fx_new = np.linalg.norm(Fx_new)
        
        # Backtracking line search
        while norm_Fx_new > norm_Fx * (1 - 1e-4 * alpha) and alpha > 1e-12:
            alpha *= 0.5
            if time_left() < 5.0:
                break
            x_new = x + alpha * s
            Fx_new = F(x_new)
            norm_Fx_new = np.linalg.norm(Fx_new)
            
        # Check if we made progress
        if norm_Fx_new < norm_Fx:
            # Update Broyden
            y = Fx_new - Fx
            # J = J + (y - J s) s^T / (s^T s)
            J = J + np.outer(y - J @ s, s) / (s @ s)
            
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
            
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x.copy()
                
            if norm_Fx <= 1e-8:
                return x
        else:
            # No progress, restart with finite difference Jacobian again
            # or break if stuck
            if i > 10:
                break
            J = compute_jacobian(x, Fx)
            
    # If we didn't converge, return the best found
    if best_norm < np.linalg.norm(F(best_x)):
        return best_x
    return x
# EVOLVE-BLOCK-END
