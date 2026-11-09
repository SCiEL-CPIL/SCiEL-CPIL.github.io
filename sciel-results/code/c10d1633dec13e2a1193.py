"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with stagnation detection and dense initial sampling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Dense initial sampling to find good basins
        # Katsuura has oscillatory structure; sample near origin and at moderate distances
        initial_points = []
        # Points near origin (sigma=0.8)
        for _ in range(30):
            initial_points.append(rng.normal(0, 0.8, n))
        # Points at moderate distance (sigma=2)
        for _ in range(30):
            initial_points.append(rng.normal(0, 2.0, n))
        # Points at larger distance (sigma=3)
        for _ in range(10):
            initial_points.append(rng.normal(0, 3.0, n))
        # A few random points across the box
        for _ in range(10):
            initial_points.append(rng.uniform(-4, 4, n))
        # Origin itself
        initial_points.append(np.zeros(n))

        # Evaluate all initial points
        for x0 in initial_points:
            if time.time() > deadline:
                break
            x0_clipped = np.clip(x0, -5, 5)
            fx0 = func(x0_clipped)
            update_best(x0_clipped, fx0)

        # Phase 2: CMA-ES from the best initial point
        # Use remaining budget for CMA-ES optimization
        used_budget = min(len(initial_points), total_budget)
        remaining_budget = total_budget - used_budget

        if remaining_budget > 20 and best_x is not None:
            self._cma_es(func, best_x.copy(), best_fx, remaining_budget, deadline, update_best)

        return best_x, best_fx

    def _cma_es(self, func, x0, f0, budget, deadline, update_best):
        """CMA-ES implementation with stagnation detection."""
        n = self.dim
        rng = np.random.default_rng(123)

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))  # lambda
        pop_size = min(pop_size, 12)  # Cap for efficiency
        sigma = 0.5  # Initial step size
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Adaptation parameters
        cmu = 1 / (4 * n + 4)
        cphi = 1 / (n + 4)
        c1 = (2 + 2 / (n + 1)) / ((n + 2) ** 2 + 2)
        cmu = min(0.4 * (1 - c1), (2 + 2 / (n + 1)) / ((n + 2) ** 2 + 2))
        chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        x_best = x0.copy()
        f_best = f0
        mean = x0.copy()
        calls_used = 0
        no_improve_count = 0
        stagnation_limit = 30  # Restart if no improvement for 30 generations

        for gen in range(budget // pop_size + 1):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                if time.time() > deadline or calls_used >= budget:
                    break
                z = rng.standard_normal(n)
                y = mean + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < f_best:
                f_best = offspring[0][0]
                x_best = offspring[0][1].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Check stagnation
            if no_improve_count >= stagnation_limit:
                break

            # Weighted mean
            mu = pop_size // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / (weights ** 2).sum()

            mean_old = mean.copy()
            mean = sum(weights[i] * offspring[i][1] for i in range(mu))

            # Update step size (CSA - Cumulative Step Adaptation)
            ps = (1 - cphi) * ps + np.sqrt(cphi * (2 - cphi) * mu_eff) / sigma * (C @ (mean - mean_old))
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cphi) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi else 0

            # Update covariance matrix
            delta = hsig * (1 - c1) * (1 + cmu * (mu_eff - 1))
            C = (1 - c1 * delta) * C
            for i in range(mu):
                y_i = (mean_old - offspring[i][1]) / sigma
                C += c1 * cmu * weights[i] * np.outer(y_i, y_i)

            # Update sigma
            sigma *= np.exp(cphi / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution paths
            pc = (1 - c1) * pc + hsig * c1 * (mean - mean_old) / sigma
            ps = (1 - cphi) * ps + hsig * np.sqrt(cphi * (2 - cphi) * mu_eff) * (mean - mean_old) / sigma

        # Final perturbation refinement
        if calls_used < budget and time.time() < deadline:
            self._refine(func, x_best, f_best, budget - calls_used, deadline, update_best)

    def _refine(self, func, x0, f0, budget, deadline, update_best):
        """Local refinement using small perturbations around the best point."""
        n = self.dim
        rng = np.random.default_rng(777)
        calls_used = 0

        sigma = 0.1
        for i in range(budget):
            if time.time() > deadline:
                break
            y = np.clip(x0 + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_used += 1
            if fy < f0:
                x0 = y
                f0 = fy
                sigma *= 0.95
            else:
                sigma *= 0.9

            if sigma < 1e-8:
                break

        update_best(x0, f0)
# EVOLVE-BLOCK-END
