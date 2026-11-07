"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with restarts and adaptive step size for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def count_func(x):
            self._call_count += 1
            fx = func(x)
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        def run_es(mu, lam, sigma_init, max_iter):
            """Run (mu+lambda)-ES with adaptive step size."""
            # Initialize population
            pop = np.array([np.clip(rng.uniform(-5, 5, dim), -5, 5) for _ in range(mu)])
            pop_f = np.array([count_func(x) for x in pop])

            # Sort initial population
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            sigma = sigma_init
            best_x = pop[0].copy()
            best_f = pop_f[0]
            no_improve = 0

            for iteration in range(max_iter):
                # Check deadline
                if time.time() > deadline:
                    break

                # Generate offspring
                offspring = np.zeros((lam, dim))
                for i in range(lam):
                    parent = pop[i % mu]
                    child = parent + sigma * rng.standard_normal(dim)
                    offspring[i] = np.clip(child, -5, 5)

                # Evaluate offspring
                off_f = np.array([count_func(x) for x in offspring])

                # Combine parents and offspring
                combined = np.vstack([pop, offspring])
                combined_f = np.concatenate([pop_f, off_f])

                # Sort and select best mu
                order = np.argsort(combined_f)
                pop = combined[order[:mu]]
                pop_f = combined_f[order[:mu]]

                # Update best
                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Adaptive step size: simple 1/5 rule approximation
                # Count successes among the best mu
                successes = sum(1 for i in range(mu) if pop_f[i] < pop_f[i + mu] if i + mu < len(pop_f))
                if successes > mu / 5:
                    sigma *= 1.3
                elif successes < mu / 5:
                    sigma *= 0.8
                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 10.0)

            return best_x, best_f

        # Strategy: Run multiple ES instances with different parameters
        # First, do a quick exploration to get a good starting point
        x0 = np.clip(rng.uniform(-5, 5, dim), -5, 5)
        f0 = count_func(x0)

        # Try a few different configurations
        configs = [
            (10, 30, 2.0, 100),
            (5, 20, 1.0, 150),
            (15, 45, 3.0, 80),
        ]

        best_x = x0
        best_f = f0

        for config in configs:
            if self._call_count >= budget or time.time() > deadline:
                break
            mu, lam, sigma_init, max_iter = config
            bx, bf = run_es(mu, lam, sigma_init, max_iter)
            if bf < best_f:
                best_x, best_f = bx, bf

        # If we still have budget, do a local refinement with (1+1)-ES
        remaining = budget - self._call_count
        if remaining > 10 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(min(remaining, 100)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = count_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
            if fx < best_f:
                best_x, best_f = x, fx

        return best_x, best_f
# EVOLVE-BLOCK-END
