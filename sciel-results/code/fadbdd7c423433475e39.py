"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # Global best tracking
        best_x = None
        best_f = np.inf

        # Phase 1: Diversified initial sampling to find good starting points
        # Use a small number of random starts to explore the space
        n_starts = min(10, max(5, total_budget // 50))
        calls_used = 0

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        candidates = []

        # Center and some spread points
        candidates.append(np.zeros(dim))
        for i in range(n_starts - 1):
            # Mix of different scales
            if i < 3:
                x0 = rng.uniform(-4, 4, dim)
            elif i < 6:
                x0 = rng.uniform(-2, 2, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            candidates.append(x0)

        # Evaluate all candidates
        for x0 in candidates:
            if calls_used >= total_budget:
                break
            fx = func(x0)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Phase 2: Run CMA-ES from the best starting point
        if calls_used >= total_budget:
            return best_x, best_f

        remaining_budget = total_budget - calls_used
        x = best_x.copy()
        fx = best_f

        # CMA-ES parameters
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # population size
        sigma = 1.0  # initial step size

        # CMA-ES state
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)

        # Precompute constants
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Track best
        best_x_es = x.copy()
        best_f_es = fx

        budget_left = remaining_budget
        iters = 0

        while budget_left > 0:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((lam, dim))
            # Apply C inverse square root (approximate using Cholesky of C)
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback: identity
                C_inv_sqrt = np.eye(dim)

            # Generate candidates
            pop = np.array([x + sigma * (C_inv_sqrt @ b) for b in B])
            # Clip to bounds
            pop = np.clip(pop, -5, 5)

            # Evaluate
            evals_this_iter = min(lam, budget_left)
            if evals_this_iter <= 0:
                break

            # Evaluate only as many as we can afford
            fvals = np.full(lam, np.inf)
            for i in range(evals_this_iter):
                fvals[i] = func(pop[i])
                budget_left -= 1
                calls_used += 1
                if fvals[i] < best_f_es:
                    best_f_es = fvals[i]
                    best_x_es = pop[i].copy()
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Weighted mean
            x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update covariance matrix
            y = (x_mean - x) / sigma
            p_s = (1 - c_sigma) * p_s + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            h_sig = (np.linalg.norm(p_s) / np.sqrt(1 - (1 - c_sigma)**(iters + 1))) < \
                    (1 + 2 / (dim + 1)) * chi_n

            p_c = (1 - c_c) * p_c + h_sig * c_c * y
            rank_mu = np.outer(p_s, p_s)
            rank_one = np.zeros((dim, dim))
            if h_sig:
                rank_one = np.outer(p_c, p_c)

            C = (1 + c_1 * (rank_one - C) + c_mu * rank_mu) * C
            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_s) / chi_n - 1))
            sigma = max(sigma, 1e-10)

            # Update position
            x = x_mean

            iters += 1

            # Check if we've made no progress recently - consider restart
            if iters % 20 == 0 and best_f_es == fx and budget_left > 10:
                # Try a small random perturbation
                x = best_x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                sigma = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
