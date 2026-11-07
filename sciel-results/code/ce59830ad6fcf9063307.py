import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        calls = 0
        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration to find promising basins
        # Rastrigin has local minima at integer coordinates in the rotated frame.
        # We sample a set of diverse points to find the best basin.
        
        n_init = 40
        candidates = []

        # Always try the origin
        eval_func(np.zeros(dim))

        for _ in range(n_init - 1):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            candidates.append((f, x))

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Multi-start CMA-ES with dynamic budget allocation
        # For top K candidates, run CMA-ES with a limited budget
        top_k = 8
        max_local_budget = 60
        min_local_budget = 20

        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            if i >= len(candidates):
                break

            f_start, x_start = candidates[i]

            x = x_start.copy()
            f_x = f_start

            # CMA-ES Parameters
            sigma = 0.5  # Initial step size
            
            # Set up CMA-ES with a specific budget
            cma_budget = min(max_local_budget, budget - calls)
            
            # Use a wrapper to count calls and respect the deadline
            cma_calls = 0
            
            # We'll use a simple loop approach to avoid complex CMA-ES termination
            # and instead do a population-based update manually if needed,
            # but since scipy CMA-ES is available, let's use it carefully.
            
            # However, scipy CMA-ES doesn't easily stop on external call limits.
            # So we'll stick to a manual (1+1)-ES or (1+lambda)-ES for safety.
            
            # Let's use a robust (1+1)-ES with adaptive sigma and stagnation handling.
            window = 10
            successes = 0
            stagnation = 0
            max_stagnation = 15
            local_calls = 0
            current_budget = min_local_budget

            while calls < budget and time.time() < deadline and local_calls < max_local_budget:
                # Early exit if this candidate is clearly worse than global best
                if local_calls >= current_budget and f_x > best_f + 0.5:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                local_calls += 1

                if f_y < f_x:
                    x = y
                    f_x = f_y
                    successes += 1
                    stagnation = 0
                    # Increase budget if improving
                    if local_calls % 10 == 0:
                        current_budget = min(max_local_budget, current_budget + 10)
                else:
                    stagnation += 1

                # Adapt sigma based on success rate
                if successes % window == 0 and successes > 0:
                    sr = successes / window
                    if sr > 0.5:
                        sigma *= 1.1
                    elif sr < 0.3:
                        sigma *= 0.9
                    sigma = np.clip(sigma, 0.01, 5.0)
                    successes = 0

                # Stagnation handling: reset sigma to escape local minimum
                if stagnation > max_stagnation:
                    sigma = max(sigma, 0.5)
                    stagnation = 0

        # Phase 3: Final Polish
        # Fine-grained search around the best point found
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.05
            remaining = budget - calls
            max_polish_steps = min(remaining, 50)

            for _ in range(max_polish_steps):
                if calls >= budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                if f_y < f_x:
                    x = y
                    f_x = f_y
                    sigma *= 0.9
                else:
                    sigma *= 0.95
                sigma = max(sigma, 0.001)

        return best_x, best_f
