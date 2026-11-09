import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_x = None
        best_f = np.inf

        # Phase 1: Global Search using Sobol quasi-random sequence
        # We use a fixed number of points for global exploration.
        # For dim=5, 64 points is a reasonable start.
        n_global = 64
        if n_global < budget:
            sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
            # Generate 2^k points. 64 = 2^6
            points = sobol.random(n_global)
            # Map [0, 1]^d to [-5, 5]^d
            points = points * 10.0 - 5.0

            for x in points:
                if time.time() > deadline:
                    break
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        else:
            # Fallback if budget is very small
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used = 1
            best_f = fx
            best_x = x.copy()

        # Phase 2: Local Refinement using CMA-ES
        # We implement a simplified CMA-ES or use scipy's if available?
        # The task says SciPy is available. However, implementing a lightweight CMA-ES 
        # or using a robust variant of (1+1)-ES with adaptive sigma is often faster 
        # and safer for small budgets than full CMA-ES overhead.
        # Let's stick to a robust (1+1)-ES but with better restart logic 
        # and potentially multiple local runs if budget allows.
        
        # Actually, let's implement a simple but effective local search:
        # Adaptive Step Size (1+1)-ES with stagnation detection.
        # We will run this until budget is exhausted or time runs out.
        
        # Initialize local search from the best global point
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Start with a moderate step size
        stagnation = 0
        stagnation_threshold = 20
        max_stagnation = 50
        local_calls = 0
        
        # We reserve some budget for potential additional local restarts if needed
        # but primarily focus on refining the current best.
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1
            local_calls += 1
            
            if fy < fx:
                x = y.copy()
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                stagnation = 0
                # Increase step size slightly on success
                sigma *= 1.2
            else:
                stagnation += 1
                # Decrease step size on failure
                sigma *= 0.9
                
            # Stagnation-based restart logic
            if stagnation >= stagnation_threshold:
                # If we are stuck, try a new local start near the global best
                # or a random point if we haven't tried many.
                # For now, let's just reset sigma and try a different direction.
                # A more aggressive restart:
                if local_calls > 50: # Only restart if we've made some progress
                     # Try a point interpolated between current and global best
                     # Or just a random perturbation of global best
                     x = best_x + 0.5 * sigma * rng.standard_normal(dim)
                     x = np.clip(x, -5, 5)
                     fx = func(x)
                     calls_used += 1
                     if fx < best_f:
                         best_f = fx
                         best_x = x.copy()
                     sigma = 2.0
                     stagnation = 0
                     stagnation_threshold = min(max_stagnation, stagnation_threshold + 10)
            
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f
