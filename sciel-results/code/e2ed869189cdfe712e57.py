import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = float('inf')
        budget_used = 0

        # Phase 1: Lattice exploitation + Sobol-like coverage for Büche-Rastrigin structure
        # The function has periodic local minima at integer coordinates in the hidden frame.
        # We sample integer lattice points and structured random points to find good basins.
        
        # Evaluate origin first
        f_val = func(np.zeros(dim))
        budget_used += 1
        best_f = f_val
        best_x = np.zeros(dim)

        # Sample integer lattice points in [-4, 4]^5
        # 9^5 = 59049 total, but we sample strategically
        n_lattice = min(150, budget // 5)
        for _ in range(n_lattice):
            if budget_used >= budget or not time_ok():
                break
            pt = rng.integers(-4, 5, size=dim).astype(float)
            f_val = func(pt)
            budget_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # Sample structured continuous points (quasi-random)
        n_structured = min(100, budget // 5)
        for _ in range(n_structured):
            if budget_used >= budget or not time_ok():
                break
            pt = rng.uniform(-5, 5, dim)
            f_val = func(pt)
            budget_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # Phase 2: Multi-restart (1+1)-ES with adaptive sigma and stagnation detection
        # Use multiple restarts from diverse seeds to escape local minima
        
        n_restarts = 8
        evals_per_restart = (budget - budget_used) // n_restarts
        if evals_per_restart < 10:
            n_restarts = 2
            evals_per_restart = (budget - budget_used) // n_restarts

        for restart in range(n_restarts):
            if budget_used >= budget or not time_ok():
                break

            # Initialize current point
            if restart == 0:
                x = best_x.copy()
            else:
                # Restart from a diverse seed: mix of best point and random
                if rng.random() < 0.5:
                    x = best_x + rng.normal(0, 1.5, dim)
                else:
                    x = rng.uniform(-4, 4, dim)
                x = np.clip(x, -5, 5)
                f_val = func(x)
                budget_used += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()

            fx = f_val if restart > 0 else best_f
            if restart == 0:
                fx = best_f
                x = best_x.copy()

            sigma = 1.0
            min_sigma = 1e-5
            max_sigma = 3.0
            no_improve_count = 0
            max_no_improve = 15

            for _ in range(evals_per_restart):
                if budget_used >= budget or not time_ok():
                    break

                # Generate offspring
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                budget_used += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(max_sigma, sigma * 1.2)
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(min_sigma, sigma * 0.85)
                    no_improve_count += 1

                # Stagnation: restart this lineage with a smaller step or break
                if no_improve_count >= max_no_improve:
                    # Try a few more with smaller sigma
                    for _ in range(3):
                        if budget_used >= budget or not time_ok():
                            break
                        y = x + max(min_sigma, sigma * 0.3) * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        budget_used += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                    break

        # Final refinement: one more (1+1)-ES pass with very small steps
        remaining = budget - budget_used
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(remaining, 50)):
                if not time_ok():
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                budget_used += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(0.5, sigma * 1.1)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(1e-6, sigma * 0.9)

        return best_x, best_f
