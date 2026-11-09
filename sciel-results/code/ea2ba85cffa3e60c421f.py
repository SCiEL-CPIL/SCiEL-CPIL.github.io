import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        total_budget = self.budget
        
        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > total_budget:
                raise ValueError("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Random exploration to find a good starting region
        init_budget = int(total_budget * 0.15)
        for _ in range(init_budget):
            if time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, n)
            try:
                wrapped_func(x)
            except (TimeoutError, ValueError):
                break

        # Phase 2: Ridge-aware optimization using BFGS with line search
        # For sharp ridge functions, we want to quickly align with the ridge direction
        # then perform rapid 1D optimization along it
        
        x_current = best_x.copy()
        
        # Estimate gradient using central differences
        def estimate_gradient(x, h=1e-4):
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                try:
                    fp = wrapped_func(x_plus)
                    fm = wrapped_func(x_minus)
                    grad[i] = (fp - fm) / (2 * h)
                except (TimeoutError, ValueError):
                    return None
            return grad
        
        # BFGS implementation for ridge optimization
        # We'll use a simplified approach: 
        # 1. Estimate gradient
        # 2. Use conjugate gradient or steepest descent with adaptive step size
        # 3. When we detect we're on a ridge (gradient is small in most directions),
        #    switch to 1D optimization along the dominant direction
        
        # Adaptive step size optimization
        step_size = 1.0
        min_step = 1e-6
        max_step = 10.0
        
        # Track gradient history to detect ridge alignment
        prev_grad = None
        
        while calls_made[0] < total_budget:
            if time.time() > deadline:
                break
            
            # Estimate gradient
            try:
                grad = estimate_gradient(x_current)
            except (TimeoutError, ValueError):
                break
            
            if grad is None:
                break
            
            # Check if we're on a ridge: gradient norm is small
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-8:
                # Likely at minimum or on flat ridge
                break
            
            # Determine search direction
            # If previous gradient exists, use conjugate direction
            if prev_grad is not None:
                # Conjugate gradient direction
                beta = np.dot(grad, grad) / max(np.dot(prev_grad, prev_grad), 1e-10)
                direction = -grad + beta * prev_grad
            else:
                # Steepest descent
                direction = -grad
            
            # Normalize direction
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                break
            direction = direction / dir_norm
            
            # Line search to find optimal step size
            best_x_ls = x_current.copy()
            best_f_ls = wrapped_func(x_current)
            
            # Try several step sizes
            for _ in range(5):
                if time.time() > deadline:
                    break
                if calls_made[0] >= total_budget:
                    break
                
                x_try = x_current + step_size * direction
                x_try = np.clip(x_try, -5, 5)
                
                try:
                    f_try = wrapped_func(x_try)
                except (TimeoutError, ValueError):
                    break
                
                if f_try < best_f_ls:
                    best_f_ls = f_try
                    best_x_ls = x_try.copy()
                    step_size = min(step_size * 1.5, max_step)
                else:
                    step_size = max(step_size * 0.5, min_step)
                    if step_size <= min_step:
                        break
            
            # Update current point
            if best_f_ls < f_current if 'f_current' in dir() else True:
                x_current = best_x_ls
                # Update global best
                if best_f_ls < best_f:
                    best_f = best_f_ls
                    best_x = best_x_ls.copy()
            
            # Update gradient history
            prev_grad = grad
            
            # If step size is very small, we might be stuck
            if step_size < min_step * 2:
                # Try a random perturbation to escape
                if np.random.rand() < 0.1:
                    x_current = x_current + np.random.normal(0, 0.5, n)
                    x_current = np.clip(x_current, -5, 5)
                    prev_grad = None
                    step_size = 1.0
        
        # Phase 3: Final polish with small perturbations
        if calls_made[0] < total_budget:
            polish_budget = total_budget - calls_made[0]
            for _ in range(min(polish_budget, 50)):
                if time.time() > deadline:
                    break
                if calls_made[0] >= total_budget:
                    break
                
                # Small random perturbation around best
                x_try = best_x + np.random.normal(0, 0.1, n)
                x_try = np.clip(x_try, -5, 5)
                try:
                    wrapped_func(x_try)
                except (TimeoutError, ValueError):
                    break

        return best_x, best_f
