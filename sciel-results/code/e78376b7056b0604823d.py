"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with aggressive restarts for multimodal BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls = 0

        # CMA-ES parameters - smaller population for faster exploration
        lambda_pop = 8
        mu = lambda_pop // 2  # number of parents
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 + 2 * min(w[0], 0.5) * (mu_eff - 1) / (n + 2)) / (
            (n + 1.3) ** 2 + mu_eff
        )
        c_mu = min(
            1 - c_1,
            2 * (
                (sqrt_mu_eff := np.sqrt(mu_eff)) * (w[0] + w[1] - 2) / (n + 2)
            ),
        )
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        def cma_es(x0, sigma0, max_evals):
            nonlocal calls
            x_mean = x0.astype(float).copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Generate and evaluate population
            def sample_and_eval():
                nonlocal calls
                pop = []
                for _ in range(lambda_pop):
                    z = rng.standard_normal(n)
                    # Use eigendecomposition for stability
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    sqrt_eigvals = np.sqrt(eigvals)
                    L = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    pop.append((y, fy))
                pop.sort(key=lambda t: t[1])
                return pop

            if calls + lambda_pop > max_evals:
                # Not enough budget for a full population, just return x0
                return x0, func(x0)

            pop = sample_and_eval()
            best_x = pop[0][0].copy()
            best_f = pop[0][1]

            evals_used = lambda_pop
            no_improve = 0

            while evals_used < max_evals:
                # Update mean
                x_parents = np.array([pop[i][0] for i in range(mu)])
                x_mean_new = np.sum(w[:, None] * x_parents, axis=0)

                # Compute C^{-1/2} using eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
                inv_L = eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T

                # y_i = C^{-1/2} (x_i - x_mean) / sigma
                y_parents = inv_L @ ((x_parents - x_mean).T / sigma)  # (n, mu)

                # Update ps
                hsig = (
                    np.linalg.norm(ps)
                    / np.sqrt(1 - (1 - c_sigma) ** (2 * (max_evals - evals_used + lambda_pop)))
                    < (1 + 2 / (n + 1)) * chi_n
                )
                ps = (1 - c_sigma) * ps + hsig * np.sqrt(
                    c_sigma * (2 - c_sigma) * mu_eff
                ) * inv_L @ ((x_mean_new - x_mean) / sigma)

                # Update pc
                pc = (
                    1 - c_c
                ) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma

                # Update C
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (
                    (1 - c_1 - c_mu) * C
                    + c_1 * (n + 2 - c_1) * C
                    - c_1 * pc[:, None] @ pc[None, :]
                    + c_mu * np.sum(
                        w[:, None, None] * y_parents[:, :, None] * y_parents[:, None, :], axis=1
                    )
                    - c_1 * delta_hsig * pc[:, None] @ pc[None, :]
                )
                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                # Project to PSD
                eigvals_C, eigvecs_C = np.linalg.eigh(C)
                eigvals_C = np.maximum(eigvals_C, 1e-12)
                C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T

                # Update sigma
                sigma *= np.exp(
                    c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1)
                )
                sigma = np.clip(sigma, 1e-10, 1e10)

                x_mean = x_mean_new

                # Evaluate new population
                pop = sample_and_eval()
                evals_used += lambda_pop

                if pop[0][1] < best_f - 1e-12:
                    best_f = pop[0][1]
                    best_x = pop[0][0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Restart condition - more aggressive: restart after 8 generations without improvement
                if no_improve >= 8 or sigma < 1e-8:
                    break

            return best_x, best_f

        # Multi-start CMA-ES with aggressive restarts
        best_x_overall = None
        best_f_overall = np.inf

        # Phase 1: Broad exploration with more random starts
        n_starts = 6
        evals_per_start = max((total_budget // n_starts) - 20, 25)

        for s in range(n_starts):
            if calls >= total_budget:
                break

            # Random start with varied sigma - start smaller for faster convergence
            x0 = rng.uniform(-4, 4, n)
            sigma0 = 1.0 + 0.3 * s

            remaining = total_budget - calls
            if remaining < lambda_pop:
                break

            max_evals = min(evals_per_start, remaining)

            x_best, f_best = cma_es(x0, sigma0, max_evals)

            if f_best < best_f_overall:
                best_f_overall = f_best
                best_x_overall = x_best

        # Phase 2: Refine around the best found point
        if calls < total_budget and best_x_overall is not None:
            remaining = total_budget - calls
            if remaining >= lambda_pop:
                # Try refinement with decreasing sigma
                for sigma_try in [0.5, 0.3, 0.15]:
                    if calls >= total_budget:
                        break
                    remaining = total_budget - calls
                    if remaining < lambda_pop:
                        break
                    x_best, f_best = cma_es(best_x_overall.copy(), sigma_try, remaining)
                    if f_best < best_f_overall:
                        best_f_overall = f_best
                        best_x_overall = x_best

        if best_x_overall is None:
            best_x_overall = np.zeros(n)
            best_f_overall = func(best_x_overall)
            calls += 1

        return best_x_overall, best_f_overall
# EVOLVE-BLOCK-END
