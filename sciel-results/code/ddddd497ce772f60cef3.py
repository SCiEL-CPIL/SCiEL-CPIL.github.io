"""Rewritten optimizer for BBOB f13 (Sharp Ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline - 5.0

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                return 1e10
            if calls >= budget:
                return 1e10
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = float(func(x_clipped))
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        rng = np.random.default_rng(12345)

        # Phase 1: Population-based evolutionary strategy
        # Maintain a population of 50 candidates with adaptive sigma
        pop_size = 50
        sigma = 2.0
        sigma_min = 1e-5
        sigma_max = 4.0
        momentum = 0.7
        
        # Initialize population: mix of structured and random points
        population = []
        fvals = []
        
        # Add some structured starting points
        structured_starts = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            np.ones(dim) * 3.0,
            -np.ones(dim) * 3.0,
            np.array([5.0] * dim),
            np.array([-5.0] * dim),
            np.array([0.0, 3.0, -3.0, 2.0, -2.0]),
            np.array([-2.0, 0.0, 2.0, -3.0, 3.0]),
        ]
        
        for s in structured_starts:
            s = np.clip(s, -5, 5)
            f = wrapped_func(s)
            population.append(s)
            fvals.append(f)
            if f < best_f:
                best_f = f
                best_x = s.copy()
        
        # Fill rest with random points
        while len(population) < pop_size and calls < budget - 10:
            x = rng.uniform(-4, 4, dim)
            x = np.clip(x, -5, 5)
            f = wrapped_func(x)
            population.append(x)
            fvals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Track success rate for adaptive sigma
        success_window = []
        window_size = 10
        velocity = np.zeros(dim)
        
        # Evolutionary loop
        generations = 0
        max_generations = 200
        
        while time_ok() and calls < budget - 10 and generations < max_generations:
            generations += 1
            fvals_arr = np.array(fvals)
            population_arr = np.array(population)
            
            # Sort by fitness
            order = np.argsort(fvals_arr)
            population = [population_arr[i] for i in order]
            fvals = [fvals_arr[i] for i in order]
            
            # Keep best half, generate new half
            keep_count = pop_size // 2
            new_population = population[:keep_count]
            new_fvals = fvals[:keep_count]
            
            # Generate new candidates from best parents
            for _ in range(pop_size - keep_count):
                if calls >= budget - 5:
                    break
                
                # Select parent: tournament selection
                candidates = rng.choice(len(population), size=min(3, len(population)), replace=False)
                parent_idx = candidates[np.argmin([fvals[i] for i in candidates])]
                parent = population[parent_idx]
                
                # Gaussian perturbation with adaptive sigma
                noise = rng.standard_normal(dim)
                
                # Add momentum component
                if np.linalg.norm(velocity) > 1e-8:
                    dir_unit = velocity / np.linalg.norm(velocity)
                    noise = 0.8 * noise + 0.2 * dir_unit * np.linalg.norm(noise)
                
                child = parent + sigma * noise
                child = np.clip(child, -5, 5)
                f_child = wrapped_func(child)
                
                if f_child < fvals[parent_idx]:
                    velocity = momentum * velocity + (1 - momentum) * (child - parent)
                    success_window.append(1)
                else:
                    velocity = momentum * velocity
                    success_window.append(0)
                
                new_population.append(child)
                new_fvals.append(f_child)
                
                if f_child < best_f:
                    best_f = f_child
                    best_x = child.copy()
            
            population = new_population
            fvals = new_fvals
            
            # Adapt sigma based on success rate
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window)
                if success_rate > 0.6:
                    sigma = min(sigma * 1.1, sigma_max)
                elif success_rate < 0.3:
                    sigma = max(sigma * 0.85, sigma_min)
                success_window = []
            
            # Occasionally inject a random point to maintain diversity
            if generations % 10 == 0 and calls < budget - 5:
                x = rng.uniform(-4, 4, dim)
                x = np.clip(x, -5, 5)
                f = wrapped_func(x)
                # Replace worst if better
                worst_idx = np.argmax(fvals)
                if f < fvals[worst_idx]:
                    population[worst_idx] = x
                    fvals[worst_idx] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
            
            # Check convergence
            fvals_arr = np.array(fvals)
            if np.max(fvals_arr) - np.min(fvals_arr) < 1e-10:
                break
        
        # Phase 2: Fine-tuning with smaller sigma around best point
        if time_ok() and calls < budget - 5:
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            velocity = np.zeros(dim)
            
            for _ in range(min(100, budget - calls - 2)):
                if not time_ok() or calls >= budget:
                    break
                
                noise = rng.standard_normal(dim)
                if np.linalg.norm(velocity) > 1e-8:
                    dir_unit = velocity / np.linalg.norm(velocity)
                    noise = 0.85 * noise + 0.15 * dir_unit * np.linalg.norm(noise)
                
                y = x + sigma * noise
                y = np.clip(y, -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    velocity = 0.6 * velocity + 0.4 * (y - x)
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.05, 0.5)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    velocity = 0.6 * velocity
                    sigma = max(sigma * 0.95, 1e-5)
                
                if sigma < 1e-6:
                    break
        
        # Phase 3: Very fine polishing
        if time_ok() and calls < budget:
            sigma = 0.02
            x = best_x.copy()
            fx = best_f
            
            for _ in range(min(50, budget - calls)):
                if not time_ok() or calls >= budget:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                
                sigma *= 0.95
                if sigma < 1e-7:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
