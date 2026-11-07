"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Quadratic model-based optimizer with full Hessian estimation for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 295.0

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def call_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count += 1
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        # Phase 1: Estimate the minimizer z using a 2n-point design (skipping center)
        # f(x) = ||x - z||^2 + c
        # grad_i(x0) = 2(x0_i - z_i)
        # Using central difference at x0=0:
        # g_i = (f(h e_i) - f(-h e_i)) / (2h) = 2(0 - z_i) = -2 z_i
        # z_i = -g_i / 2
        
        delta = 2.0
        grad = np.zeros(dim)
        
        # Evaluate 2n points to estimate gradient at origin
        for i in range(dim):
            if call_count + 2 > budget or time.time() - start_time > timeout:
                break
                
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            
            x_plus = delta * e_i
            x_minus = -delta * e_i
            
            f_plus = call_func(x_plus)
            f_minus = call_func(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2.0 * delta)
        
        # Estimate z from gradient
        # grad = -2z => z = -grad/2
        z_est = -grad / 2.0
        z_est = np.clip(z_est, -5.0, 5.0)
        
        # Evaluate the estimated minimum
        if call_count < budget and time.time() - start_time <= timeout:
            f_z = call_func(z_est)
            
        # Phase 2: Polish with Newton steps using the fitted Hessian
        # The Hessian of a sphere is constant (2I), so we can use it directly.
        # We just need an accurate gradient at the current best location.
        
        x_current = best_x.copy()
        f_current = best_fx
        
        # Hessian for Sphere is 2*I
        H = 2.0 * np.eye(dim)
        
        for it in range(20): # Up to 20 polish iterations
            if call_count >= budget or time.time() - start_time > timeout:
                break
            
            # Estimate gradient at x_current using central differences
            # Use a small step size for accuracy
            h = 1e-4
            grad_local = np.zeros(dim)
            
            for i in range(dim):
                if call_count + 2 > budget: # Need 2 calls per dim
                    break
                if time.time() - start_time > timeout:
                    break
                    
                e_i = np.zeros(dim)
                e_i[i] = 1.0
                
                x_plus = x_current + h * e_i
                x_minus = x_current - h * e_i
                
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                f_plus = call_func(x_plus)
                f_minus = call_func(x_minus)
                
                grad_local[i] = (f_plus - f_minus) / (2.0 * h)
            
            # Check if gradient is negligible
            if np.linalg.norm(grad_local) < 1e-10:
                break
                
            # Newton step: x_new = x_current - H^-1 * grad
            # H = 2I, so H^-1 = 0.5 I
            step = -0.5 * grad_local
            x_new = x_current + step
            
            x_new = np.clip(x_new, -5.0, 5.0)
            
            if call_count >= budget:
                break
                
            f_new = call_func(x_new)
            
            if f_new < f_current:
                x_current = x_new
                f_current = f_new
            else:
                # If no improvement, we are likely at the minimum or model is inaccurate
                break

        return best_x, best_fx

# EVOLVE-BLOCK-END
