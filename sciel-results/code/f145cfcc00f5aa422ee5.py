"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES-like optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            calls += 1
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Multi-start: evaluate several initial points to get good initial best
        # Weierstrass f16 has global min near origin but shifted/rotated
        # Try center and a few random points
        init_points = [np.zeros(dim)]
        for _ in range(9):
            init_points.append(rng.uniform(-4, 4, dim))
        
        for ip in init_points:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(ip)

        # Now do local search from best point using adaptive step size
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        stagnation = 0
        max_stagnation = 30
        window_size = 10
        successes = []
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x, fx = y, fy
                successes.append(1)
                stagnation = 0
            else:
                successes.append(0)
                stagnation += 1
            
            if len(successes) > window_size:
                successes.pop(0)
            
            if len(successes) == window_size:
                rate = sum(successes) / window_size
                if rate > 0.25:
                    sigma *= 1.15
                elif rate < 0.1:
                    sigma *= 0.85
            
            # Restart if stuck
            if stagnation > max_stagnation:
                # Try a new random restart far from current best
                if rng.random() < 0.5:
                    x = rng.uniform(-4, 4, dim)
                    fx = evaluate(x)
                else:
                    x = best_x + 0.5 * rng.standard_normal(dim)
                    fx = evaluate(x)
                sigma = 0.5
                stagnation = 0
                successes = []
        
        return best_x, best_f
# EVOLVE-BLOCK-END
