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
        # f(x) = ||Rx - z||^2 + c
        # grad(x) = 2 R^T (Rx - z)
        # At x=0, grad(0) = -2 R^T z
        # Wait, the previous logic assumed f(x) = ||x-z||^2. 
        # If there is a rotation R, f(x) = ||Rx - z||^2.
        # grad(x) = 2 R^T (Rx - z).
        # Hessian H = 2 R^T R = 2 I.
        # So Newton step is x - 0.5 * grad(x).
        
        # To find the minimizer, we need to solve R^T(Rx - z) = 0 => Rx = z => x = R^T z.
        # We don't know R or z.
        # However, the Newton method with H=2I will converge to the minimizer 
        # provided we start close enough.
        
        # Initial guess: 
        # If we assume R=I, z = -0.5 * grad(0).
        # If R != I, this guess is not the minimizer, but it is a point in the domain.
        # Let's evaluate grad at 0 to get a direction towards the minimum?
        # Actually, for a quadratic form with positive definite Hessian, 
        # the Newton method converges globally if the function is convex.
        # Sphere is convex.
        # So we can just start at x=0 and run Newton.
        # But a good initial guess helps AOCC (early progress).
        
        # Let's estimate the "effective" shift.
        # If we evaluate f at +/- delta * e_i, we can estimate the gradient at 0.
        # grad_i(0) approx (f(delta e_i) - f(-delta e_i)) / (2 delta)
        
        delta = 2.0
        grad_0 = np.zeros(dim)
        
        for i in range(dim):
            if call_count + 2 > budget or time.time() - start_time > timeout:
                break
                
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            
            x_plus = delta * e_i
            x_minus = -delta * e_i
            
            f_plus = call_func(x_plus)
            f_minus = call_func(x_minus)
            
            grad_0[i] = (f_plus - f_minus) / (2.0 * delta)
        
        # If R=I, z = -0.5 * grad_0.
        # If R!=I, this is not the minimizer, but let's use it as a starting point.
        # Actually, a better starting point might be 0 itself if the shift is small,
        # or the estimated z if the shift is large.
        # Let's evaluate the estimated z and 0, and pick the better one.
        
        z_est = -0.5 * grad_0
        z_est = np.clip(z_est, -5.0, 5.0)
        
        f_z = call_func(z_est)
        
        x_current = best_x.copy()
        f_current = best_fx
        
        # Phase 2: Newton Refinement
        # H = 2I
        # Step = -0.5 * grad(x)
        
        # We have plenty of budget. Let's run until budget is low or convergence.
        # 2n calls for gradient + 1 call for step = 11 calls per iteration.
        # Budget 500. Phase 1 used ~11 calls. Remaining ~489.
        # Max iterations ~44.
        
        max_iterations = 40
        
        for it in range(max_iterations):
            if call_count >= budget or time.time() - start_time > timeout:
                break
            
            # Estimate gradient at x_current
            # Use a small step size for accuracy, but not too small to avoid numerical noise
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
            
            # Check convergence
            grad_norm = np.linalg.norm(grad_local)
            if grad_norm < 1e-8:
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
                # If no improvement, we might be at the minimum or noise.
                # Try a smaller step? Or break.
                # For Sphere, Newton should always improve unless at min.
                # If it doesn't, we are likely at machine precision limit.
                break

        return best_x, best_fx

# EVOLVE-BLOCK-END
