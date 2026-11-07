import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f23 (Katsuura) on [-5,5]^5.
    Uses a hybrid approach: initial random sampling + CMA-ES-like adaptive search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_point(x):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or check_time():
                return
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        rng = np.random.default_rng(2024)

        # Phase 1: Initial exploration - sample diverse points
        # Katsuura has a nested structure; the minimum is at a specific point
        # Try origin and axis-aligned points first
        try_point(np.zeros(d))
        
        # Sample some structured points
        for sign in [-1, 1]:
            for i in range(d):
                if check_time() or used_calls >= budget:
                    break
                x = np.zeros(d)
                x[i] = sign * 3.0
                try_point(x)
        
        # Random sampling to explore the space
        n_init = min(80, budget - used_calls)
        for _ in range(n_init):
            if used_calls >= budget or check_time():
                break
            x = rng.uniform(-5, 5, d)
            try_point(x)

        # Phase 2: CMA-ES-like adaptive search
        if best_x is not None and used_calls < budget and not check_time():
            # Initialize CMA-ES-like parameters
            mean = best_x.copy().astype(float)
            sigma = 1.0
            pop_size = min(15, max(6, d + 2))  # population size
            remaining = budget - used_calls
            
            # CMA-ES-like adaptive search
            for gen in range(remaining // pop_size + 1):
                if used_calls >= budget or check_time():
                    break
                
                # Generate population
                offspring = []
                for k in range(pop_size):
                    if used_calls >= budget or check_time():
                        break
                    # Standard normal perturbation
                    z = rng.standard_normal(d)
                    x_new = np.clip(mean + sigma * z, -5, 5)
                    try_point(x_new)
                    if best_x is not None:
                        offspring.append((best_f, best_x.copy()))
                
                if len(offspring) < 2:
                    break
                
                # Update mean using weighted average of best offspring
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                # Use top half to update mean
                n_top = max(1, len(offspring) // 2)
                top = [x for _, x in offspring[:n_top]]
                new_mean = np.mean(top, axis=0)
                
                # Update sigma based on improvement
                old_mean_f = func(mean) if used_calls < budget else best_f
                new_mean_f = min(f for f, _ in offspring[:n_top])
                
                if new_mean_f < old_mean_f:
                    # Converging, keep or slightly reduce sigma
                    sigma *= 0.95
                else:
                    # Not improving, increase sigma
                    sigma *= 1.1
                
                # Clip sigma to reasonable range
                sigma = np.clip(sigma, 0.001, 5.0)
                
                # Update mean
                mean = new_mean
                
                # If we found a significantly better point, reset around it
                if offspring[0][0] < best_f - 1e-6:
                    mean = offspring[0][1].copy()
                    sigma = max(sigma * 0.5, 0.01)

        # Phase 3: Fine-tuning with small perturbations
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls
            
            # Try decreasing step sizes
            for scale in [0.5, 0.1, 0.01, 0.001, 0.0001]:
                if used_calls >= budget or check_time():
                    break
                n_attempts = min(30, remaining)
                for _ in range(n_attempts):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    y = np.clip(x + scale * z, -5, 5)
                    try_point(y)
                    if best_x is not None and best_f < fx:
                        x = best_x.copy()
                        fx = best_f

        return best_x, best_f
