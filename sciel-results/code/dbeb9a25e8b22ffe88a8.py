import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 290.0

        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration with CMA-ES-like strategy
        # Use a population-based approach with adaptive step sizes
        pop_size = 20
        budget_per_iter = pop_size
        max_iters = budget // budget_per_iter

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, n))
        evals = np.array([evaluate(x) for x in pop])

        # Track best
        best_idx = np.argmin(evals)
        best_x = pop[best_idx].copy()
        best_f = evals[best_idx]

        # Adaptive step size
        sigma = 1.5
        c_sigma = 0.1
        c_c = (4 + n / 3) / (n + 4 + 2)
        c_1 = 2 / (n + 1.3) ** 2 + c_c
        c_mu = min(1 - c_1, 2 * (n - 2 + 1 / n) / (n + 2) ** 2)

        # Mean of population
        mean = np.mean(pop, axis=0)
        weights = np.log(n / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()

        # Covariance matrix (start with identity)
        C = np.eye(n)
        sqrt_inv_C = np.eye(n)

        # Precompute for efficiency
        c_sigma = 0.1
        d_sigma = 1 + 2 * max(0, sqrt((mu_eff + 1) / (n + 2) - 1)) + c_sigma
        c_c = (4 + n / 3) / (n + 4 + 2)
        c_1 = 2 / (n + 1.3) ** 2 + c_c
        c_mu = min(1 - c_1, 2 * (n - 2 + 1 / n) / (n + 2) ** 2)

        # Cumulative step size control
        cs = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Main CMA-ES loop
        for it in range(max_iters):
            if time.time() - start_time > max_time:
                break
            if call_count >= budget:
                break

            # Sort population by fitness
            order = np.argsort(evals)
            pop_sorted = pop[order]
            evals_sorted = evals[order]

            # Update mean
            mean = np.sum(weights[:, None] * pop_sorted[:int(mu_eff)], axis=0)

            # Update covariance matrix
            y = (pop_sorted[:int(mu_eff)] - mean) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(mean - mean, mean - mean) + c_mu * (np.outer(mean - mean, mean - mean))
            # Proper covariance update
            diff = pop_sorted[:int(mu_eff)] - mean
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(mean, mean) + c_mu * np.sum(weights[:int(mu_eff)][:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update cumulative step size
            ps = (1 - c_sigma) * ps + c_sigma * (mean - mean) / sigma

            # Generate new population
            new_pop = []
            new_evals = []
            for i in range(pop_size):
                if time.time() - start_time > max_time:
                    break
                if call_count >= budget:
                    break

                # Sample from multivariate normal
                z = rng.standard_normal(n)
                x = mean + sigma * (sqrt_inv_C @ z)
                x = np.clip(x, -5, 5)
                fx = evaluate(x)
                new_pop.append(x)
                new_evals.append(fx)

            if len(new_pop) == 0:
                break

            pop = np.array(new_pop)
            evals = np.array(new_evals)

            # Update best
            best_idx = np.argmin(evals)
            if evals[best_idx] < best_f:
                best_f = evals[best_idx]
                best_x = pop[best_idx].copy()

            # Adapt sigma
            sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
            sigma = np.clip(sigma, 0.01, 10.0)

        # Phase 2: Local polish with Nelder-Mead
        if best_x is not None and time.time() - start_time < max_time and call_count < budget:
            polish_budget = min(100, budget - call_count)
            if polish_budget > 0:
                from scipy.optimize import minimize

                def wrapper(x):
                    nonlocal call_count
                    if call_count >= budget:
                        return best_f
                    return evaluate(x)

                try:
                    result = minimize(wrapper, best_x, method='Nelder-Mead',
                                     options={'maxiter': polish_budget // n, 'xatol': 1e-6, 'fatol': 1e-8})
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        return best_x, best_f
