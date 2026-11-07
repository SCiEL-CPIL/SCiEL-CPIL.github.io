"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu, lambda)-ES with adaptive step size and restarts for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None
        self._deadline = time.time() + 280  # 300s limit, 20s safety margin

        def count_func(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        # Initial point
        x = rng.uniform(-4, 4, self.dim)
        fx = count_func(x)

        mu = 4
        lam = 16
        sigma = 1.5

        # Main (mu, lambda)-ES loop
        while self._call_count < self.budget:
            # Check deadline
            if time.time() > self._deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                if self._call_count >= self.budget:
                    break
                # Use a mixture: some large steps, some small steps
                if i < lam // 4:
                    step = sigma * 2.0
                elif i < lam // 2:
                    step = sigma
                else:
                    step = sigma * 0.5
                y = np.clip(x + step * rng.standard_normal(self.dim), -5, 5)
                fy = count_func(y)
                offspring.append((fy, y))

            if len(offspring) == 0:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Take top mu
            new_x = offspring[0][1]
            new_fx = offspring[0][0]

            # Adaptive step size based on success rate
            successes = sum(1 for fy, _ in offspring if fy < fx)
            rate = successes / len(offspring)
            if rate > 0.2:
                sigma *= 1.2
            elif rate < 0.1:
                sigma *= 0.8
            else:
                sigma *= 1.0

            # Clamp sigma
            sigma = max(0.01, min(sigma, 10.0))

            # Update parent
            x, fx = new_x, new_fx

            # Check for stagnation - restart if no improvement in recent iterations
            if self._call_count > 50 and self._best_f == fx and self._call_count % 100 == 0:
                # Soft restart: perturb current best
                x = self._best_x + 0.5 * rng.standard_normal(self.dim)
                x = np.clip(x, -5, 5)
                fx = count_func(x)
                sigma = 0.5

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
