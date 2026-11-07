"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style Schwefel f20 with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s kill
        budget = self.budget
        n = self.dim

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            f = float(func(x_clipped))
            calls_used += 1
            return x_clipped, f

        def cma_es(x0, f0, max_evals):
            """Run CMA-ES starting from x0 with initial fitness f0."""
            nonlocal best_x, best_f, calls_used

            # CMA-ES parameters
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2  # number of selected individuals

            # Weights for recombination
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Adaptation rates
            cc = (4 + n / lam) / (n + 4 + 2 * n / lam)
            cs = (lam + 2 + 2 / (n + lam)) / (n + lam + 5 + 2 * lam / n)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) / (n + 4)
            cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2) ** 2 + mueff))

            # Initialize
            mean = x0.copy().astype(float)
            sigma = 1.0
            C = np.eye(n)
            C_sqrt_inv = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Track best
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Store initial point
            archive = [(x0.copy(), f0)]

            for gen in range(max_evals // lam + 1):
                if time.time() > deadline or calls_used >= budget:
                    break

                # Sample population
                pop = []
                pop_f = []
                for i in range(lam):
                    if time.time() > deadline or calls_used >= budget:
                        break
                    # Sample from N(mean, sigma^2 * C)
                    z = np.random.standard_normal(n)
                    x_sample = mean + sigma * (C_sqrt_inv @ z)
                    x_sample = np.clip(x_sample, -5.0, 5.0)
                    res = evaluate(x_sample)
                    if res is None:
                        return
                    x_s, f_s = res
                    pop.append(x_s)
                    pop_f.append(f_s)
                    if f_s < best_f:
                        best_f = f_s
                        best_x = x_s.copy()

                if len(pop) < 2:
                    break

                # Sort by fitness
                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = [pop_f[i] for i in order]

                # Update mean
                new_mean = sum(weights[i] * pop[i] for i in range(mu))

                # Update step size (sigma)
                y_sigma = (new_mean - mean) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (gen + 1)) / chi_n < 1.4 + 2 / (n + 1))
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(sigma, 1e-12)

                # Update evolution path
                y_c = (new_mean - mean) / sigma
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * y_c

                # Update covariance matrix
                delta_h = (1 + h_sigma) * cc * (2 - cc) - 1
                outer = np.outer(y_c, y_c)
                C = (1 + c1 * delta_h - c1 - cmu) * C + cmu * sum(weights[i] * np.outer(pop[i] - mean, pop[i] - mean) for i in range(mu)) + c1 * (delta_h * np.outer(p_c, p_c) + (1 - (1 - cc) ** 2 * (1 + delta_h)) * np.eye(n))

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-14)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C = (C + C.T) / 2
                    C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt_inv = np.eye(n)

                mean = new_mean
                mean = np.clip(mean, -5.0, 5.0)

                # Convergence check
                if sigma < 1e-12 or np.all(np.abs(mean - pop[0]) < 1e-10):
                    break

            # Local refinement after CMA-ES
            if best_x is not None and calls_used < budget and time.time() < deadline:
                self._local_refine(func, best_x, best_f)

        def _local_refine(self, func, x_best, f_best):
            """Coordinate-wise local search to polish the solution."""
            nonlocal calls_used, best_x, best_f
            dim = self.dim
            sigma = 0.1
            for _ in range(50):
                if time.time() > deadline or calls_used >= budget:
                    break
                improved = False
                for i in range(dim):
                    for direction in [-1, 1]:
                        x_try = x_best.copy()
                        x_try[i] += direction * sigma
                        x_try = np.clip(x_try, -5.0, 5.0)
                        res = evaluate(x_try)
                        if res is None:
                            return
                        x_try, f_try = res
                        if f_try < f_best - 1e-15:
                            x_best = x_try
                            f_best = f_try
                            improved = True
                            if f_best < best_f:
                                best_f = f_best
                                best_x = x_best.copy()
                            break
                    if improved:
                        break
                if not improved:
                    sigma *= 0.5
                    if sigma < 1e-8:
                        break

        # Main CMA-ES run with diverse restarts
        rng = np.random.default_rng(42)

        # Strategy: one main CMA-ES run, then restarts if stagnation
        # First, try from origin (known Schwefel minimum region)
        x0 = np.zeros(n)
        res = evaluate(x0)
        if res is None:
            return best_x, best_f
        x0, f0 = res
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        remaining = budget - calls_used
        max_evals = max(10, min(300, remaining))
        cma_es(x0, f0, max_evals)

        # If we still have budget, do a second CMA-ES from a different start
        if best_x is not None and calls_used < budget and time.time() < deadline:
            x0_2 = rng.uniform(-3.0, 3.0, n)
            res = evaluate(x0_2)
            if res is not None:
                x0_2, f0_2 = res
                if f0_2 < best_f:
                    best_f = f0_2
                    best_x = x0_2.copy()
                remaining = budget - calls_used
                max_evals = max(10, min(200, remaining))
                cma_es(x0_2, f0_2, max_evals)

        # Final polish
        if best_x is not None and calls_used < budget and time.time() < deadline:
            self._local_refine(func, best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
