"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cmaes(self, func, rng, x0, fx0, calls_made, best_f, best_x, deadline,
               budget, dim, lambda_pop, max_generations, sigma0):
        """Run CMA-ES from x0. Returns (best_x, best_f, calls_made)."""
        n = dim
        lam = lambda_pop
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        Cinv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        best_f_local = fx0
        best_x_local = x0.copy()
        gen = 0

        for gen in range(max_generations):
            if calls_made >= budget or time.time() >= deadline:
                break

            # Generate lambda samples
            B = np.linalg.cholesky(C)
            y_samples = []
            x_samples = []
            f_samples = []

            for i in range(lam):
                z = rng.standard_normal(n)
                y = B @ z
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = func(x_new)
                calls_made += 1
                y_samples.append(y)
                x_samples.append(x_new)
                f_samples.append(f_new)

                if f_new < best_f_local:
                    best_f_local = f_new
                    best_x_local = x_new.copy()
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()

            # Sort by fitness
            order = np.argsort(f_samples)
            y_sorted = np.array([y_samples[i] for i in order])
            x_sorted = np.array([x_samples[i] for i in order])
            f_sorted = np.array([f_samples[i] for i in order])

            # Update mean
            y_mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            x_mean = x_mean + sigma * y_mean
            x_mean = np.clip(x_mean, -5.0, 5.0)

            # Update covariance
            ys = y_sorted[:mu]
            diffmat = ys - y_mean[None, :]
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma) ** 2) * np.eye(n)) + c_mu * np.sum(weights[:, None, None] * diffmat[:, :, None] * diffmat[:, None, :], axis=0)

            # Update step size
            ps = (1 - c_sigma) * ps + c_sigma / chi_n * np.linalg.norm(y_mean) * (1 if np.dot(y_mean, np.linalg.cholesky(C) @ np.linalg.inv(np.linalg.cholesky(C)) @ y_mean) > 0 else -1)
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update pc
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            pc = (1 - c_c) * pc + h_sigma * c_c * y_mean

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                Cinv = np.eye(n)

            # Stopping criteria
            if sigma < 1e-8 or np.max(np.diag(C)) / np.min(np.diag(C)) > 1e14:
                break

            # If no improvement in last 3 generations, break
            if gen > 3 and f_sorted[0] >= f_sorted[0] and gen >= 10:
                # Check if we've made little progress
                pass

        return best_x_local, best_f_local, calls_made, best_f, best_x

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0  # 30s safety margin

        rng = np.random.default_rng(42)

        best_f = np.inf
        best_x = np.zeros(dim)
        calls_made = 0

        # Multiple restarts with CMA-ES
        n_restarts = 4
        lambda_pop = 12
        max_gens_per_restart = [40, 30, 20, 15]  # Decreasing budget per restart

        start_points = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-5, 5, dim),
        ]

        for r in range(n_restarts):
            if calls_made >= budget or time.time() >= deadline:
                break

            x0 = start_points[r]
            fx0 = func(x0)
            calls_made += 1

            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            if calls_made >= budget or time.time() >= deadline:
                break

            remaining = budget - calls_made
            max_gens = min(max_gens_per_restart[r], remaining // lambda_pop)
            if max_gens < 2:
                break

            sigma0 = 2.0 if r == 0 else 1.5

            best_x, best_f, calls_made, best_f, best_x = self._cmaes(
                func, rng, x0, fx0, calls_made, best_f, best_x,
                deadline, budget, dim, lambda_pop, max_gens, sigma0
            )

        # Final polish with small (1+1)-ES
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            sigma = 0.2
            x = best_x.copy()
            fx = best_f
            for _ in range(min(remaining, 30)):
                if time.time() >= deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.85
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
