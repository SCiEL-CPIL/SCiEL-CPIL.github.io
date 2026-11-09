"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Stochastic (1+lambda)-ES with adaptive step size and restarts for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility, but random enough to avoid bad corners
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # (1+lambda)-ES parameters
        lambda_ = 20  # Offspring count
        sigma = 1.0   # Initial step size
        sigma_min = 1e-4
        sigma_max = 5.0
        target_success_rate = 0.2
        
        # Initialize best point with a random point
        current_x = rng.uniform(-5, 5, dim)
        current_f = wrapped_func(current_x)
        
        if current_f < best_f:
            best_f = current_f
            best_x = current_x.copy()
            
        no_improve_count = 0
        restart_interval = 50 # Restart if no improvement for 50 generations (1000 calls approx)
        
        while call_count < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            offspring_f = []
            
            for i in range(lambda_):
                if call_count + 1 >= budget or time.time() > deadline:
                    break
                z = rng.normal(0, 1, dim)
                x_child = current_x + sigma * z
                x_child = np.clip(x_child, -5, 5)
                f_child = wrapped_func(x_child)
                offspring.append(x_child)
                offspring_f.append(f_child)
            
            if not offspring:
                break
                
            # Find best offspring
            idx_best = np.argmin(offspring_f)
            best_offspring_x = offspring[idx_best]
            best_offspring_f = offspring_f[idx_best]
            
            # Check for improvement
            if best_offspring_f < current_f:
                # Success
                current_x = best_offspring_x
                current_f = best_offspring_f
                no_improve_count = 0
                
                # Adapt sigma: increase if success rate is high
                success_rate = 1.0 / lambda_ # Only 1 best is tracked, so this is a proxy. 
                # Better adaptation: use the number of successful offspring relative to lambda
                # But we only kept the best. Let's use a simpler 1/5th rule approximation.
                # If we improved, likely many did. Let's just nudge up slightly if we improved significantly?
                # Standard (1+lambda) often uses 1/5th rule.
                # Let's use a simple heuristic: if improved, increase sigma slightly, else decrease.
                # But Rastrigin needs careful sigma.
                # Let's try: if improved, sigma *= 1.1, else sigma *= 0.9
                sigma *= 1.1
            else:
                no_improve_count += 1
                sigma *= 0.9
                
            # Clamp sigma
            sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Restart logic: If stuck for too long, restart from a random point or near best
            if no_improve_count > restart_interval:
                # Restart: jump to a random point, but maybe keep some memory of best?
                # For Rastrigin, full restart is often better to escape deep local minima.
                # Let's restart from a random point in the box.
                current_x = rng.uniform(-5, 5, dim)
                current_f = wrapped_func(current_x)
                no_improve_count = 0
                sigma = 1.0 # Reset sigma
                
        return best_x, best_f

# EVOLVE-BLOCK-END
