"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: CMA-ES global search followed by BFGS local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES implementation for global search
        # Phase 1: CMA-ES with ~70% of budget
        cma_budget = int(budget * 0.7)

        # Initialize
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # CMA-ES parameters
        mu = max(1, dim // 2)
        lambda_pop = 2 * mu
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Boundary handling
        def clip(x):
            return np.clip(x, -5, 5)

        # Generate initial population
        def sample_population(x, sigma):
            pop = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = clip(y)
                pop.append(y)
            return np.array(pop)

        # Evaluate initial population
        pop = sample_population(x, sigma)
        fits = np.array([func(p) for p in pop])
        order = np.argsort(fits)
        best_idx = order[0]
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]

        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f

        for gen in range(cma_budget // lambda_pop):
            if time.time() > deadline:
                break

            # Select
            selected = pop[order[:mu]]
            selected_fits = fits[order[:mu]]

            # Update mean
            new_x = np.sum(weights[:, None] * selected, axis=0)
            new_x = clip(new_x)

            # Update step size control variables
            x_bar = pop[order[:mu]]
            d_mean = (new_x - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - c_sigma)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * d_mean
            hsig_c = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - c_c)

            # Update covariance matrix
            delta_d = (1 - hsig_c) * hsig_c * (1 + d_sigma) * np.outer(pc, pc)
            C = (1 - c1 * c1) * C + c1 * (delta_d + np.sum(weights[:, None, None] * np.outer(selected - x, selected - x), axis=0))
            c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

            # Sample new population
            pop = np.array([clip(new_x + sigma * (L @ rng.standard_normal(dim))) for _ in range(lambda_pop)])
            fits = np.array([func(p) for p in pop])
            order = np.argsort(fits)
            best_idx = order[0]
            if fits[best_idx] < global_best_f:
                global_best_x = pop[best_idx].copy()
                global_best_f = fits[best_idx]

            x = new_x

        # Phase 2: BFGS local refinement with remaining budget
        remaining_budget = budget - cma_budget

        # Use scipy's BFGS with function evaluation budget tracking
        from scipy.optimize import minimize

        eval_count = [0]
        max_bfgs_evals = max(10, remaining_budget)

        def objective_with_count(x):
            if time.time() > deadline:
                raise Exception("timeout")
            eval_count[0] += 1
            if eval_count[0] > max_bfgs_evals:
                raise Exception("budget exceeded")
            return func(np.clip(x, -5, 5))

        # Try BFGS from the best point found
        try:
            result = minimize(
                objective_with_count,
                global_best_x,
                method='BFGS',
                options={'maxiter': 100, 'maxfun': max_bfgs_evals, 'gtol': 1e-8}
            )
            if result.success and result.fun < global_best_f:
                global_best_x = np.clip(result.x, -5, 5)
                global_best_f = result.fun
        except Exception:
            pass

        # If we still have budget, do a few more CMA-ES steps or local search
        if time.time() < deadline - 10:
            # Simple local refinement with decreasing step sizes
            for step_scale in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
                if time.time() > deadline:
                    break
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction)
                    trial = np.clip(global_best_x + step_scale * direction, -5, 5)
                    trial_f = func(trial)
                    if trial_f < global_best_f:
                        global_best_x = trial
                        global_best_f = trial_f

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
