import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        deadline = start_time + 280  # safety margin before 300s limit

        # For Sphere function: f(x) = ||Q(x-s) + d||^2 + o
        # We can use gradient-based optimization since we can estimate gradients
        # via finite differences. This is much more efficient for smooth convex functions.
        
        # Phase 1: Use Nelder-Mead to find a good region
        # Phase 2: Use gradient descent with line search to refine
        
        calls = 0
        best_x = np.zeros(n)
        best_f = float('inf')
        
        # Try multiple starting points for Nelder-Mead
        np.random.seed(42)  # Deterministic for reproducibility
        start_points = [
            np.zeros(n),
            np.ones(n) * 2,
            -np.ones(n) * 2,
            np.random.randn(n) * 3,
            np.random.randn(n) * 3,
        ]
        
        for sp in start_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Run Nelder-Mead from this starting point
            result_x, result_f, calls_used = self._nelder_mead(
                func, sp, n, budget - calls, deadline
            )
            calls += calls_used
            
            if result_f < best_f:
                best_f = result_f
                best_x = result_x.copy()
        
        # Phase 2: Gradient-based refinement from the best point
        if calls < budget and time.time() < deadline:
            result_x, result_f, calls_used = self._gradient_descent(
                func, best_x, n, budget - calls, deadline
            )
            calls += calls_used
            
            if result_f < best_f:
                best_f = result_f
                best_x = result_x.copy()
        
        return best_x, best_f

    def _nelder_mead(self, func, x0, n, budget, deadline):
        """Run Nelder-Mead simplex method."""
        calls = 0
        best_x = x0.copy()
        best_f = float('inf')
        
        # Initialize simplex
        alpha = 1.5  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Create initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            x_i = x0.copy()
            x_i[i] += 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
            simplex.append(x_i)
        
        # Evaluate simplex
        f_values = np.zeros(n + 1)
        for i in range(n + 1):
            if calls >= budget or time.time() > deadline:
                break
            f_values[i] = func(simplex[i])
            calls += 1
            if f_values[i] < best_f:
                best_f = f_values[i]
                best_x = simplex[i].copy()
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f, calls
        
        # Main loop
        max_iter = 100
        for _ in range(max_iter):
            if calls >= budget or time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(f_values)
            simplex = [simplex[i].copy() for i in order]
            f_values = f_values[order]
            
            # Check convergence
            if np.max(f_values) - np.min(f_values) < 1e-10:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            if calls >= budget or time.time() > deadline:
                break
            f_r = func(x_r)
            calls += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
            
            if f_r < f_values[0]:
                # Expansion
                if calls >= budget or time.time() > deadline:
                    break
                x_e = centroid + gamma * (x_r - centroid)
                f_e = func(x_e)
                calls += 1
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls >= budget or time.time() > deadline:
                                break
                            f_values[i] = func(simplex[i])
                            calls += 1
                            if f_values[i] < best_f:
                                best_f = f_values[i]
                                best_x = simplex[i].copy()
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if calls >= budget or time.time() > deadline:
                            break
                        f_values[i] = func(simplex[i])
                        calls += 1
                        if f_values[i] < best_f:
                            best_f = f_values[i]
                            best_x = simplex[i].copy()
        
        return best_x, best_f, calls

    def _gradient_descent(self, func, x0, n, budget, deadline):
        """Run gradient descent with line search."""
        calls = 0
        x = x0.copy()
        f = func(x)
        calls += 1
        best_x = x.copy()
        best_f = f
        
        # Estimate gradient using finite differences
        eps = 1e-5
        lr = 0.1
        max_iter = 100
        
        for _ in range(max_iter):
            if calls >= budget or time.time() > deadline:
                break
            
            # Compute gradient
            grad = np.zeros(n)
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = func(x_plus)
                calls += 1
                
                if calls >= budget or time.time() > deadline:
                    break
                x_minus = x.copy()
                x_minus[i] -= eps
                f_minus = func(x_minus)
                calls += 1
                
                grad[i] = (f_plus - f_minus) / (2 * eps)
            
            if calls >= budget or time.time() > deadline:
                break
            
            # Check if gradient is small
            if np.linalg.norm(grad) < 1e-8:
                break
            
            # Line search
            best_lr = lr
            best_f_line = f
            for lr_try in [lr, lr * 0.5, lr * 0.25, lr * 0.1]:
                if calls >= budget or time.time() > deadline:
                    break
                x_new = x - lr_try * grad
                f_new = func(x_new)
                calls += 1
                
                if f_new < best_f_line:
                    best_f_line = f_new
                    best_lr = lr_try
                    best_x_line = x_new.copy()
            
            if best_f_line < best_f:
                x = best_x_line
                f = best_f_line
                best_f = f
                best_x = x.copy()
            else:
                # No improvement, reduce learning rate
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        return best_x, best_f, calls
