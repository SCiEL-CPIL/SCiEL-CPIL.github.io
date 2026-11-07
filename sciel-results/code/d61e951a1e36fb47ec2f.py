"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES-like adaptive covariance optimizer with restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff / (n + 1.5) + c_sigma) / ((n + 1.8) * c_sigma + 2)
        c_mu = (2 * (mu_eff / (n + 1.8) - c_1)) / ((n + 1.8) * c_sigma + 2)
        c_mu = min(c_mu, 2 - c_1)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            return fx

        # Initial population spread
        sigma0 = 2.0

        while calls_made < budget:
            # Restart logic: if we've made some progress, use current best as center
            if best_x is None:
                center = rng.uniform(-3, 3, n)
            else:
                center = best_x.copy()

            # Initialize CMA-ES state
            mean = center.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            inv_sqrt_C = np.eye(n)

            # Precompute Cholesky of C
            try:
                B, logdet = np.linalg.slogdet(C)
                L = np.linalg.cholesky(C)
                inv_sqrt_C = np.linalg.inv(L.T)
            except np.linalg.LinAlgError:
                inv_sqrt_C = np.eye(n)

            # Evaluate initial population
            pop = []
            fvals = []
            for i in range(min(lambda_, budget - calls_made)):
                z = rng.standard_normal(n)
                x = np.clip(mean + sigma * z, -5, 5)
                fx = evaluate(x)
                pop.append(x)
                fvals.append(fx)

            if calls_made >= budget:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # Update mean
            mean_new = sum(weights[i] * pop[i] for i in range(mu))

            # Update evolution paths
            y = (mean_new - mean) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C @ y

            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made + 1))) / chi_n < 1.4 + 2 / (n + 1))

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            rank_mu = np.outer(pc, pc) + delta_hsig * C
            rank_one = np.outer(ps, ps)
            C = (1 - c_sigma * (c_1 + c_mu - c_1)) * C + c_1 * rank_mu + c_mu * rank_one

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 50)

            mean = mean_new

            # Check for stagnation or convergence
            if np.linalg.norm(mean_new - mean) < 1e-10:
                break

            # If population has converged too much, break to restart
            if len(set(map(tuple, [tuple(np.round(p, 6)) for p in pop[:min(3, len(pop))]]))) == 1:
                break

            # Check if we should restart (diversity loss)
            if calls_made > budget * 0.7:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
