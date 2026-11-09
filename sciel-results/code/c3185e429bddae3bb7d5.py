"""Rewritten optimizer: CMA-ES global search + L-BFGS-B local polish for rotated Rosenbrock."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES for global valley location, L-BFGS-B for high-precision polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # safety margin before 300s kill

        # Counted function wrapper
        call_count = [0]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Track best
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.clip(x, -5, 5).copy()

        # Phase 1: CMA-ES for global exploration
        # CMA-ES adapts its covariance matrix to handle anisotropic landscapes
        # like the rotated Rosenbrock valley
        rng = np.random.default_rng(12345)

        # Initial population spread across the domain
        # Start centered at origin with reasonable sigma
        x0 = np.zeros(dim)
        sigma0 = 2.0  # initial step size

        # CMA-ES parameters
        pop_size = max(4 + int(3 * np.log(dim)), 8)  # ~13 for dim=5
        pop_size = min(pop_size, 15)

        # We'll run CMA-ES manually to control budget
        # Population
        population = [x0 + sigma0 * rng.standard_normal(dim) for _ in range(pop_size)]
        population = [np.clip(p, -5, 5) for p in population]

        # Evaluate initial population
        for x in population:
            if call_count[0] >= budget or time.time() > deadline:
                break
            f = counted_func(x)
            update_best(x, f)

        # CMA-ES iteration
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)  # covariance matrix
        C_sqrt = np.eye(dim)  # square root of C
        # Step size control
        cs = (pop_size + 2) / (dim + pop_size + 5)
        ds = 1 + 2 * max(0, (np.sqrt((pop_size + 2 - 1) / (dim + 2)) - 1) / cs)
        cmu = min(0.4, (pop_size - 2 + 1 / pop_size) / ((dim + 2) * (pop_size + 2)))
        ps = np.zeros(dim)
        ps_norm = np.sqrt(dim)

        # Track fitness history
        fitness_history = []

        max_cma_iterations = 30  # limit CMA-ES iterations to leave budget for polish

        for iteration in range(max_cma_iterations):
            if call_count[0] >= budget or time.time() > deadline:
                break

            # Generate population
            population = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x_new = mean + sigma * (C_sqrt @ z)
                x_new = np.clip(x_new, -5, 5)
                population.append(x_new)

            # Evaluate
            fitnesses = []
            for x in population:
                if call_count[0] >= budget or time.time() > deadline:
                    fitnesses.append(1e10)
                    continue
                f = counted_func(x)
                fitnesses.append(f)
                update_best(x, f)

            if len(fitnesses) < pop_size:
                break

            # Sort by fitness
            sorted_idx = np.argsort(fitnesses)
            sorted_pop = [population[i] for i in sorted_idx]
            sorted_fit = [fitnesses[i] for i in sorted_idx]

            # Update mean
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)

            old_mean = mean.copy()
            mean = np.sum([weights[i] * sorted_pop[i] for i in range(mu)], axis=0)

            # Update evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt.T @ (mean - old_mean) / sigma)

            # Update sigma
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) < (1.4 + 1 / dim) * ps_norm:
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / ps_norm - 1))
            sigma = max(sigma, 1e-10)

            # Update covariance matrix
            if iteration % 2 == 0:  # update every 2 iterations to save computation
                delta_mean = (mean - old_mean) / sigma
                C = (1 - cs * (2 - cs) - cmu) * C
                C += cs * (2 - cs) * np.outer(ps, ps)
                C += cmu * np.sum([weights[i] * np.outer(sorted_pop[i] - old_mean, sorted_pop[i] - old_mean) / (sigma ** 2) for i in range(mu)], axis=0)

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = (eigvecs * eigvals) @ eigvecs.T

                # Update square root via Cholesky with fallback
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    # Fallback: eigenvalue decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            # Early stopping: if sigma is very small or improvement is minimal
            if sigma < 1e-6:
                break
            if iteration > 5 and sorted_fit[0] > best_f * 1.001 + 1e-8:
                # No improvement for several iterations
                if iteration > 10:
                    break

        # Phase 2: L-BFGS-B local polish from the best point found
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            try:
                result = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 300, 'maxls': 50, 'gtol': 1e-10, 'ftol': 1e-15}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: If we still have budget, try a second polish from a slightly perturbed best
        if best_x is not None and call_count[0] < budget - 20 and time.time() < deadline:
            # Try a few small perturbations around best
            for _ in range(3):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                perturbation = 0.1 * rng.standard_normal(dim)
                x_perturbed = np.clip(best_x + perturbation, -5, 5)
                try:
                    result = minimize(
                        counted_func,
                        x_perturbed,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 100, 'maxls': 30, 'gtol': 1e-10}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        return best_x, best_f
