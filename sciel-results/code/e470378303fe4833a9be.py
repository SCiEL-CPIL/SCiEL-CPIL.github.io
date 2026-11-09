"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Gradient Descent with quadratic step estimation for Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Initialize at center
        x = np.zeros(dim)
        fx = func(x)
        call_count = 1
        
        best_x = x.copy()
        best_fx = fx
        
        # Gradient estimation parameters
        eps = 1e-6
        
        # Adaptive step size
        # We estimate the quadratic coefficient 'a' in f(x) ~ a*||x-c||^2
        # Optimal step alpha = 1 / (2*a)
        # Initial guess for a=1 -> alpha=0.5
        alpha = 0.5 
        a_est = 1.0
        
        # Main loop
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a central difference gradient
            if call_count + 2 * dim > budget:
                if call_count + dim <= budget:
                     # Forward difference fallback
                     grad = np.zeros(dim)
                     for i in range(dim):
                         x_p = x.copy()
                         x_p[i] += eps
                         f_p = func(x_p)
                         call_count += 1
                         grad[i] = (f_p - fx) / eps
                else:
                    break
            else:
                # Central difference
                grad = np.zeros(dim)
                for i in range(dim):
                    x_p = x.copy()
                    x_p[i] += eps
                    x_m = x.copy()
                    x_m[i] -= eps
                    f_p = func(x_p)
                    f_m = func(x_m)
                    call_count += 2
                    grad[i] = (f_p - f_m) / (2 * eps)
            
            # Check convergence
            grad_norm_sq = np.dot(grad, grad)
            if grad_norm_sq < 1e-12:
                break
            
            # Try to find a step that decreases the function
            # We use the estimated alpha, but verify it
            # If it fails, we shrink alpha
            
            accepted = False
            current_alpha = alpha
            
            # Limit inner retries to avoid wasting budget on one step
            for _ in range(5): 
                x_new = x - current_alpha * grad
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = func(x_new)
                call_count += 1
                
                if f_new < fx:
                    # Accept
                    accepted = True
                    
                    # Update estimate of 'a'
                    # Delta f = f_new - fx
                    # Delta x = -current_alpha * grad
                    # For quadratic: Delta f = -alpha * ||g||^2 + a * alpha^2 * ||g||^2
                    # a = (Delta f + alpha * ||g||^2) / (alpha^2 * ||g||^2)
                    delta_f = f_new - fx
                    denom = (current_alpha ** 2) * grad_norm_sq
                    if denom > 1e-12:
                        a_new = (delta_f + current_alpha * grad_norm_sq) / denom
                        # Clamp a_new to reasonable values to prevent instability
                        if a_new > 1e-6 and a_new < 1e6:
                            a_est = a_new
                        else:
                            # Fallback if estimate is bad
                            a_est = a_est * 0.8 if delta_f < 0 else a_est * 1.2
                    
                    # Update best
                    x = x_new
                    fx = f_new
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    
                    # Set next alpha based on new estimate
                    alpha = 1.0 / (2.0 * a_est)
                    # Cap alpha to avoid huge jumps if a_est is very small
                    alpha = min(alpha, 10.0)
                    break
                else:
                    # Reject, shrink step
                    current_alpha *= 0.5
                    if current_alpha < 1e-8:
                        break

            if not accepted:
                # If we couldn't find a step even with very small alpha, 
                # we might be at a minimum or in a flat region.
                # Or the gradient estimate is bad.
                # Break to avoid wasting budget.
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
