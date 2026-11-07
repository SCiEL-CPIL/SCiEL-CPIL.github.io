import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        start_time = time.time()
        deadline = start_time + 325.0

        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Multi-scale stochastic tunneling method
        # The Weierstrass function has self-similar structure with features at multiple scales.
        # We start with large steps to find broad basins, then progressively reduce step size
        # by an order of magnitude at each level to exploit the fractal nature of the function.

        def multiscale_tunnelling(start_x, sigma0, n_levels, budget_limit, min_sigma=1e-8):
            nonlocal calls_made, best_f, best_x

            n = dim
            x = np.clip(start_x.copy(), -5.0, 5.0)
            fx = wrapper(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = sigma0
            steps_in_level = 0
            level = 0

            while level < n_levels and calls_made < budget_limit and time.time() < deadline - 2:
                # Each level uses a fixed number of steps, then reduces sigma
                steps_per_level = max(10, int(budget_limit / n_levels))
                
                while steps_in_level < steps_per_level and calls_made < budget_limit and time.time() < deadline - 2:
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * z
                    x_trial = np.clip(x_trial, -5.0, 5.0)

                    fx_trial = wrapper(x_trial)

                    if fx_trial < best_f:
                        best_f = fx_trial
                        best_x = x_trial.copy()

                    # Accept move if better, or with some probability for exploration
                    if fx_trial < fx or (fx_trial >= fx and rng.random() < 0.1):
                        x = x_trial
                        fx = fx_trial

                    steps_in_level += 1

                # Reduce sigma by an order of magnitude for next level
                sigma /= 10.0
                if sigma < min_sigma:
                    break
                level += 1
                steps_in_level = 0

            return x, fx

        # Use multi-scale tunneling from multiple starting points
        # Start with a few diverse initial points
        n_starts = 3
        starts = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4.0, 4.0, dim))

        budget_per_start = int(total_budget * 0.6 / n_starts)
        
        for start_x in starts:
            if calls_made >= total_budget * 0.65 or time.time() > deadline - 10:
                break
            remaining = min(budget_per_start, total_budget - calls_made)
            multiscale_tunnelling(start_x, sigma0=2.0, n_levels=6, budget_limit=remaining)

        # Final polish with very small steps
        if calls_made < total_budget and time.time() < deadline - 5:
            remaining = total_budget - calls_made
            multiscale_tunnelling(best_x, sigma0=0.1, n_levels=3, budget_limit=remaining, min_sigma=1e-10)

        return best_x, best_f
