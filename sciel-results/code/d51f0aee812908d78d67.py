# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard and robust restarts for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # CMA-ES parameters tuned for dim=5, budget=300
        # Population size: small to maximize generations
        lam = 6
        mu = lam // 2  # top 50% selected

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()

        # Adaptation rates
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        c1 = (2 + 2 * min(1, 0.4 / (dim + 1))) / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        cc = 4 + dim / 4

        def run_cmaes(x0, sigma0, max_calls):
            """Run CMA-ES from x0 with initial sigma0, using at most max_calls evaluations."""
            nonlocal best_x, best_f, calls

            # Local state
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            L = np.linalg.cholesky(C)

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            local_calls = 0

            while local_calls < max_calls:
                # Check time
                if time.time() > deadline - 5:
                    break

                # Generate offspring
                offspring = []
                offspring_f = []
                for i in range(lam):
                    if local_calls + 1 > max_calls:
                        break
                    if time.time() > deadline - 5:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                    fy = func(y)
                    offspring_f.append(fy)
                    local_calls += 1
                    calls += 1

                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(offspring_f) < 2:
                    break

                # Sort by fitness
                idx = np.argsort(offspring_f)
                sorted_off = [offspring[i] for i in idx]
                sorted_f = [offspring_f[i] for i in idx]

                # Save old mean
                old_mean = x_mean.copy()

                # Update mean (weighted average of top mu)
                x_mean = np.zeros(dim)
                for i in range(mu):
                    x_mean += weights[i] * sorted_off[i]

                # Update evolution paths
                # v_mean: normalized difference between new and old mean
                if sigma > 1e-10:
                    v_mean = (x_mean - old_mean) / sigma
                else:
                    v_mean = np.zeros(dim)

                # h_sigma: indicator for sigma adaptation
                h_sigma = min(1.0, (1.0 - np.exp(-1.0 / ds)) / 2.0)

                # Update p_sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * v_mean
                sigma *= np.exp(cs * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
                sigma = np.clip(sigma, 1e-8, 50)

                # Update p_c
                h_c = min(1.0, (1.0 - np.exp(-1.0 / ds)) / 2.0)
                p_c = (1 - c1 - cmu) * p_c + h_c * np.sqrt((c1 + cmu) * (2 - c1 - cmu) * mueff) * v_mean

                # Rank-mu update for covariance
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    if sigma > 1e-10:
                        vi = (sorted_off[i] - old_mean) / sigma
                    else:
                        vi = np.zeros(dim)
                    rank_mu += weights[i] * np.outer(vi, vi)

                # Update covariance matrix
                C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * rank_mu
                # Ensure symmetry
                C = (C + C.T) / 2

                # Re-cholesky with fallback
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset covariance if not positive definite
                    C = np.eye(dim)
                    L = np.linalg.cholesky(C)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    sigma = max(sigma, 1e-3)

                # Stopping criterion: convergence
                if np.linalg.norm(x_mean - old_mean) < 1e-10 * sigma:
                    break

            return local_calls

        # Phase 1: Initial CMA-ES run with generous budget
        # Start at origin since minimizer is in [-4,4]^5
        x0 = np.zeros(dim)
        sigma0 = 2.0

        # Use most of the budget for the main run
        main_budget = int(budget * 0.85)
        run_cmaes(x0, sigma0, main_budget)

        # Phase 2: If time and budget remain, do a local refinement
        if calls < budget and time.time() < deadline - 10:
            remaining = budget - calls
            if remaining >= 10:
                # Local (1+1)-ES refinement around best point found
                x = best_x.copy()
                fx = best_f
                sigma_local = 0.5
                for _ in range(min(remaining, 50)):
                    if time.time() > deadline - 5:
                        break
                    y = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma_local *= 1.3
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                    else:
                        sigma_local *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
