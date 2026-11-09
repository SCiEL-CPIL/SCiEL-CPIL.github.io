import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Adaptive gradient-based optimization with finite difference step scaling
        # Start with a reasonable initial point and use gradient descent with adaptive step size
        
        # Initial point: center of domain
        x = np.zeros(dim)
        f_x = func(x)
        calls_made += 1
        best_x = x.copy()
        best_f = f_x
        
        # Adaptive step size parameters
        step_size = 1.0  # Initial step size
        min_step = 1e-8
        max_step = 10.0
        step_decay = 0.95  # Decay factor for step size
        step_growth = 1.5  # Growth factor when stuck
        
        # Gradient estimation via finite differences
        def estimate_gradient(x, f_x, h):
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = func(x_plus)
                calls_made += 1
                
                x_minus = x.copy()
                x_minus[i] -= h
                x_minus = np.clip(x_minus, -5, 5)
                f_minus = func(x_minus)
                calls_made += 1
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            return grad
        
        # Gradient descent loop with adaptive step size
        stuck_counter = 0
        improvement_threshold = 1e-10
        
        while calls_made < budget and not check_deadline():
            # Estimate gradient
            h = max(step_size * 0.1, 1e-6)
            grad = estimate_gradient(x, f_x, h)
            
            # Check if we've run out of budget
            if calls_made >= budget or check_deadline():
                break
            
            # Compute gradient magnitude
            grad_mag = np.linalg.norm(grad)
            
            # If gradient is very small, we're likely near optimum
            if grad_mag < 1e-10:
                stuck_counter += 1
                if stuck_counter > 5:
                    break
                continue
            
            # Adaptive step size: scale inversely with gradient magnitude
            # Large gradient -> small step, small gradient -> larger step
            adaptive_step = step_size / (1.0 + grad_mag / 10.0)
            adaptive_step = np.clip(adaptive_step, min_step, max_step)
            
            # Take a step in the negative gradient direction
            x_new = x - adaptive_step * grad
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate new point
            f_new = func(x_new)
            calls_made += 1
            
            # Check improvement
            if f_new < f_x - improvement_threshold:
                # Good improvement, update
                x = x_new
                f_x = f_new
                stuck_counter = 0
                
                # Update best
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                # Gradually reduce step size
                step_size = max(step_size * step_decay, min_step)
            else:
                # No improvement, try smaller step
                stuck_counter += 1
                step_size = max(step_size * 0.5, min_step)
                
                # If stuck for too long, try a random restart from current best
                if stuck_counter > 10:
                    # Random perturbation from best point
                    if rng.random() < 0.3:
                        x = best_x + rng.standard_normal(dim) * step_size
                        x = np.clip(x, -5, 5)
                        f_x = func(x)
                        calls_made += 1
                    stuck_counter = 0
                    step_size = max(step_size * step_growth, min_step)
        
        # Phase 2: Fine-tuning with local search if budget remains
        if calls_made < budget and not check_deadline():
            # Use a smaller step size for fine-tuning
            fine_step = min(step_size, 0.01)
            local_best_x = best_x.copy()
            local_best_f = best_f
            local_calls = 0
            max_local_calls = min(50, budget - calls_made)
            
            while local_calls < max_local_calls and not check_deadline():
                # Try small perturbations in each dimension
                improved = False
                for i in range(dim):
                    if calls_made >= budget or check_deadline():
                        break
                    
                    for direction in [-1, 1]:
                        x_try = local_best_x.copy()
                        x_try[i] += direction * fine_step
                        x_try = np.clip(x_try, -5, 5)
                        
                        f_try = func(x_try)
                        calls_made += 1
                        local_calls += 1
                        
                        if f_try < local_best_f:
                            local_best_x = x_try
                            local_best_f = f_try
                            improved = True
                            break
                    
                    if calls_made >= budget or check_deadline():
                        break
                
                if not improved:
                    break
                
                # Update global best
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
                
                # Reduce step size
                fine_step *= 0.5
                if fine_step < 1e-8:
                    break
        
        return best_x, best_f
