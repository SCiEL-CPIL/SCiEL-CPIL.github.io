"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer with finite-difference gradients for rotated Rosenbrock (f9)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def estimate_gradient(x, f0, h=1e-4):
            """Estimate gradient using central finite differences."""
            nonlocal calls
            grad = np.zeros(n)
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    return None
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                f_plus = func(x_plus)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    return None
                f_minus = func(x_minus)
                calls += 1
                grad[i] = (f_plus - f_minus) / (2 * h)
            return grad

        def gradient_descent(x0, f0, max_iter=5000):
            """Variable-step gradient descent with backtracking line search."""
            nonlocal calls, best_x, best_f
            
            x = x0.copy()
            f = f0
            step_size = 0.1  # Initial step size
            max_step = 1.0
            min_step = 1e-8
            c1 = 1e-4  # Armijo condition
            rho = 0.5   # Backtracking factor
            patience = 50
            no_improve_count = 0

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Estimate gradient
                grad = estimate_gradient(x, f)
                if grad is None:
                    break
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    break
                
                # Normalize gradient for direction
                direction = -grad / grad_norm
                
                # Backtracking line search
                alpha = step_size
                f_new = None
                for _ in range(20):
                    if time.time() > deadline or calls >= budget:
                        break
                    x_trial = x + alpha * direction
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1
                    
                    # Armijo condition
                    if f_trial < f - c1 * alpha * grad_norm:
                        f_new = f_trial
                        break
                    alpha *= rho
                
                if f_new is None:
                    # No acceptable step found; reduce step size and try again
                    step_size *= 0.5
                    if step_size < min_step:
                        break
                    continue
                
                # Update
                x = x_trial
                f = f_new
                no_improve_count = 0
                
                # Track best
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                
                # Adapt step size
                if f < f0 - 0.01 * abs(f0):
                    step_size = min(step_size * 1.5, max_step)
                else:
                    step_size *= 0.8
                
                f0 = f
                no_improve_count += 1
                
                if no_improve_count >= patience and step_size < 0.01:
                    break

            return x, f

        # Phase 1: Random sampling to find a good starting point
        n_random = min(20, budget // 10)
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, n)
            f_try = func(x_try)
            calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        # Also evaluate origin
        if calls < budget and time.time() < deadline:
            f_origin = func(np.zeros(n))
            calls += 1
            if f_origin < best_f:
                best_f = f_origin
                best_x = np.zeros(n).copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Gradient descent from best point
        x_curr, f_curr = gradient_descent(best_x.copy(), best_f)
        
        if f_curr < best_f:
            best_f = f_curr
            best_x = x_curr.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Local refinement with smaller steps
        # Try multiple small perturbations around the best point
        for restart in range(5):
            if calls >= budget or time.time() > deadline:
                break
            
            # Create a small perturbation
            if restart == 0:
                x_new = best_x + 0.1 * rng.standard_normal(n)
            elif restart == 1:
                x_new = best_x + 0.05 * rng.standard_normal(n)
            elif restart == 2:
                x_new = best_x + 0.2 * rng.standard_normal(n)
            else:
                x_new = best_x + 0.15 * rng.standard_normal(n)
            
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls += 1
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            
            # Run short gradient descent from this point
            if calls < budget and time.time() < deadline:
                x_ref, f_ref = gradient_descent(x_new, f_new, max_iter=200)
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

        # Phase 4: Final polish with very small steps
        if calls < budget and time.time() < deadline:
            x_final, f_final = gradient_descent(best_x.copy(), best_f, max_iter=100)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
