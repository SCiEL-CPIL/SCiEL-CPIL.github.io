import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        D = self.dim
        budget = self.budget
        
        call_count = [0]
        best_x = [np.zeros(D)]
        best_f = [np.inf]
        
        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count[0] >= budget:
                raise ValueError("Budget exceeded")
            
            call_count[0] += 1
            f_val = float(func(x))
            if f_val < best_f[0]:
                best_f[0] = f_val
                best_x[0] = x.copy()
            return f_val

        # 1. Initial Center Evaluation
        # Sphere functions are symmetric and convex. The center of the box is a strong prior for the global minimum location before shift/rotation is accounted for, or at least a safe starting point.
        x0 = np.zeros(D)
        try:
            f0 = wrapped_func(x0)
        except Exception:
            return (best_x[0], best_f[0])

        # 2. Finite Difference Gradient Estimation
        # We estimate the gradient at the current best point using central differences.
        # This costs 2*D evaluations per gradient estimate.
        # Since D=5, this is 10 calls.
        
        # We will iterate:
        # 1. Estimate gradient at current x.
        # 2. Perform a line search along the negative gradient (steepest descent).
        # 3. If progress is slow, we can try to estimate Hessian or just continue.
        # However, for Sphere, the gradient points directly to the minimum.
        # Actually, for Sphere f(x) = ||x - x*||^2 + c, the gradient is 2(x - x*).
        # So the minimum is at x - grad/2.
        # We can exploit this specific structure!
        
        # Let's test if we can jump directly.
        # If we assume f(x) is quadratic, we can fit a local quadratic model.
        # But a simpler exploit: 
        # If we take a step in the direction of -grad, and the function is a perfect sphere,
        # the next point x_new = x - alpha * grad.
        # The optimal alpha for a quadratic is 1/L where L is the Lipschitz constant of the gradient (eigenvalue of Hessian).
        # For Sphere f(x) = ||x - c||^2, Hessian is 2I. L=2. Optimal alpha = 1/2.
        # So x_new = x - 0.5 * grad.
        # If we use this, we should converge in 1 step IF the gradient is exact and the function is exactly quadratic with Hessian 2I.
        # The hidden shift and rotation preserve the quadratic nature. The Hessian is still 2I (rotation doesn't change the eigenvalues of 2I, shift doesn't change Hessian).
        # Wait, is the Hessian always 2I?
        # BBOB f1 is Sphere. The standard form is f(x) = ||x - x_opt||^2.
        # The transformations are x -> R(x - shift) + offset? Or similar.
        # If it's a rigid transformation, the Hessian is preserved.
        # Let's try a "Newton-like" step assuming Hessian is 2I (or proportional to I).
        
        # Strategy:
        # 1. Evaluate at x0.
        # 2. Estimate gradient g at x0 using central differences.
        # 3. Propose x1 = x0 - g / 2.0.
        # 4. Evaluate x1.
        # 5. If x1 is better, we are likely very close.
        # 6. If not, or if we want to be safe, we can do a line search.
        # But let's try the direct jump first. It's the most efficient if the model holds.
        
        def estimate_gradient(x):
            grad = np.zeros(D)
            h = 1e-4
            for i in range(D):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                # Ensure we stay in bounds? The func clips, so we don't need to clip x for the call, 
                # but the gradient might be distorted if we clip. 
                # However, the problem says "func(x) clips a finite length-5 vector to the box".
                # If x is inside [-5,5], xp and xm might go out.
                # If they go out, the clip makes them flat, so gradient component becomes 0.
                # This is a boundary issue. If we are far from boundary, it's fine.
                # Given the minimizer is in [-4,4]^5, and we start at 0, we are likely safe for the first few steps.
                
                fp = wrapped_func(xp)
                fm = wrapped_func(xm)
                grad[i] = (fp - fm) / (2 * h)
            return grad

        try:
            # Step 1: Gradient at 0
            g0 = estimate_gradient(x0)
            
            # Step 2: Newton-like step
            # x_new = x - H^{-1} g. Assuming H = 2I, x_new = x - 0.5 * g.
            x1 = x0 - 0.5 * g0
            
            # Check bounds
            x1 = np.clip(x1, -5.0, 5.0)
            
            f1 = wrapped_func(x1)
            
            # If f1 is significantly better, we might be at the minimum.
            # Let's do one more iteration of gradient estimation at x1 to see if we can refine.
            # If we are at the minimum, gradient should be ~0.
            
            # Check time/budget
            if time.time() > deadline or call_count[0] >= budget:
                return (best_x[0], best_f[0])
                
            # Step 3: Refine at x1
            g1 = estimate_gradient(x1)
            x2 = x1 - 0.5 * g1
            x2 = np.clip(x2, -5.0, 5.0)
            
            f2 = wrapped_func(x2)
            
            # Step 4: Refine at x2
            if time.time() > deadline or call_count[0] >= budget:
                return (best_x[0], best_f[0])
                
            g2 = estimate_gradient(x2)
            x3 = x2 - 0.5 * g2
            x3 = np.clip(x3, -5.0, 5.0)
            
            f3 = wrapped_func(x3)
            
            # If we still have budget, we can do a local Powell or Nelder-Mead on the best point found
            # to clean up any finite difference noise or model mismatch.
            
            remaining = budget - call_count[0]
            if remaining > 20:
                from scipy.optimize import minimize
                start_x = best_x[0].copy()
                try:
                    res = minimize(
                        wrapped_func, 
                        start_x, 
                        method='Nelder-Mead',
                        options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                except Exception:
                    pass
        except Exception:
            pass

        return (best_x[0], best_f[0])
