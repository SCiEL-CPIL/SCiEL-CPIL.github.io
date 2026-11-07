# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio search for BBOB f23 (Katsuura):
       broad random exploration, then a small population CMA-ES to exploit
       multiple local minima, followed by coordinate-wise and fine polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(np.asarray(x, dtype=float), -5.0, 5.0)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(x)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # ---------- Phase 1: broad exploration ----------
        # First point at center to anchor early progress
        evaluate(np.zeros(dim))

        sigma_explore = 2.5
        n_explore = min(50, budget // 5)
        for i in range(n_explore):
            if calls_made >= budget or remaining_time() < 5.0:
                break
            if rng.random() < 0.35:
                y = clip(rng.uniform(-5, 5, dim))
            else:
                y = clip(best_x + sigma_explore * rng.standard_normal(dim))
            fy = evaluate(y)
            if fy < best_f - 0.3:
                sigma_explore = min(3.5, sigma_explore * 1.1)
            elif i > 0 and i % 10 == 0:
                sigma_explore = max(0.8, sigma_explore * 0.93)

        # ---------- Phase 2: small population CMA-ES ----------
        # Maintain diversity to avoid premature convergence in Katsuura's
        # hyper-multimodal landscape.
        pop_size = 12
        x_mean = best_x.copy()
        sigma = 1.0
        # Initialize population around current best and a few random points
        population = []
        fvals = []
        for i in range(pop_size):
            if i == 0:
                px = x_mean.copy()
            elif i == 1:
                px = clip(x_mean + 2.0 * rng.standard_normal(dim))
            elif i == 2:
                px = clip(rng.uniform(-5, 5, dim))
            else:
                px = clip(x_mean + sigma * rng.standard_normal(dim))
            pf = evaluate(px)
            population.append(px)
            fvals.append(pf)

        # CMA-ES parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.7) + (4 + dim / (dim + 4)))
        cs = (dim + 1.7) / (dim + 9.2)
        cmu = 2 / ((dim + 1.7) ** 2 + 10)
        ps = np.zeros(dim)
        C = np.eye(dim)
        # Precompute C^{1/2} once initially, update each generation
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        # Run CMA-ES until budget or time runs out
        max_generations = 80
        for gen in range(max_generations):
            if calls_made >= budget or remaining_time() < 10.0:
                break

            # Sample offspring
            offspring = []
            off_fvals = []
            for i in range(pop_size):
                y = x_mean + sigma * (sqrt_C @ rng.standard_normal(dim))
                y = clip(y)
                fy = evaluate(y)
                offspring.append(y)
                off_fvals.append(fy)

            # Combine parent and offspring, select best pop_size
            all_x = population + offspring
            all_f = fvals + off_fvals
            order = np.argsort(all_f)
            selected_x = [all_x[i] for i in order[:pop_size]]
            selected_f = [all_f[i] for i in order[:pop_size]]

            # Update mean (weighted)
            weights = np.exp(-np.arange(pop_size) / (pop_size / 2))
            weights /= weights.sum()
            x_mean_new = np.average(selected_x, axis=0, weights=weights)
            x_mean_new = clip(x_mean_new)

            # Update evolution path and covariance
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (
                C @ (x_mean_new - x_mean) / (sigma)
            )
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 1 / dim) * chiN:
                C = (1 - cs * (2 - cs) - cmu) * C + cmu * np.outer(ps, ps) + (
                    (1 - cs * (2 - cs) - cmu) * C - C
                )  # simplified: use standard update below
                # Proper standard update:
                C = (1 - cs * (2 - cs) - cmu) * C + cmu * np.outer(ps, ps)
                # Add rank-mu update
                v = [selected_x[i] - x_mean for i in range(pop_size)]
                rank_mu = np.zeros((dim, dim))
                for i in range(pop_size):
                    rank_mu += weights[i] * np.outer(v[i], v[i])
                C = (1 - cs * (2 - cs) - cmu) * C + cmu * rank_mu
            else:
                C = (1 - cs * (2 - cs) - cmu) * C + cmu * rank_mu

            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / (dim * 2) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 0.005, 5.0)

            x_mean = x_mean_new
            population = selected_x
            fvals = selected_f

            # Update global best
            if selected_f[0] < best_f:
                best_f = selected_f[0]
                best_x = selected_x[0].copy()

            # Re-inject diversity if stagnating
            if gen % 10 == 5 and calls_made < budget and remaining_time() > 10.0:
                # Replace worst individual with a random point
                worst_idx = pop_size - 1
                new_x = clip(rng.uniform(-5, 5, dim))
                new_f = evaluate(new_x)
                population[worst_idx] = new_x
                fvals[worst_idx] = new_f

        # ---------- Phase 3: coordinate-wise refinement ----------
        if calls_made < budget and remaining_time() > 5.0:
            x = best_x.copy()
            fx = best_f
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
            for step in step_sizes:
                if calls_made >= budget or remaining_time() < 3.0:
                    break
                for coord in range(dim):
                    if calls_made >= budget or remaining_time() < 3.0:
                        break
                    for direction in [-1, 1]:
                        if calls_made >= budget or remaining_time() < 3.0:
                            break
                        y = x.copy()
                        y[coord] += direction * step
                        y = clip(y)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                        else:
                            break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---------- Phase 4: final high-precision polish ----------
        if calls_made < budget and remaining_time() > 3.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.02
            n_fine = min(100, budget - calls_made)
            for i in range(n_fine):
                if calls_made >= budget or remaining_time() < 2.0:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.03
                else:
                    sigma *= 0.90
                    if sigma < 0.001:
                        sigma = 0.001
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
