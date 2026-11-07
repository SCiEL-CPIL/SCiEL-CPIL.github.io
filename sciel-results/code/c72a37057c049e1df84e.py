"""Rewritten optimizer: CMA-ES with adaptive restarts and Rastrigin-aware initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with restarts and Rastrigin-aware initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Rastrigin-aware initialization: sample near integer points
        # where local minima are likely to be
        rng = np.random.default_rng(42)

        def rastrigin_aware_sample(n_samples):
            """Generate samples biased toward integer grid points."""
            samples = []
            for _ in range(n_samples):
                # Mix: some purely random, some near integers
                if rng.random() < 0.5:
                    x = rng.uniform(-5, 5, dim)
                else:
                    # Near integer points (Rastrigin minima are near integers)
                    x = np.round(rng.uniform(-4, 4, dim)) + rng.normal(0, 0.5, dim)
                    x = np.clip(x, -5, 5)
                samples.append(x)
            return np.array(samples)

        # Phase 1: Coarse exploration
        n_coarse = min(50, budget // 10)
        best_x = None
        best_f = np.inf

        coarse_samples = rastrigin_aware_sample(n_coarse)
        for x in coarse_samples:
            if time.time() > deadline:
                break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: CMA-ES with restarts
        remaining = budget - n_coarse
        # Split remaining budget into restarts
        max_restarts = 4
        budget_per_restart = remaining // max_restarts

        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            if remaining <= 10:
                break

            # Initialize CMA-ES around current best
            x_mean = best_x.copy()
            sigma = max(0.5, 2.0 / (restart + 1))  # decreasing sigma for later restarts

            # CMA-ES parameters
            lam = max(4, int(4 + 3 * np.log(dim)))  # population size
            mu = lam // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / weights.sum()
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 - cc + min(mu / (mueff + 2), 2) / dim) / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

            # Covariance matrix and its Cholesky factor
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Track best
            best_x_restart = best_x.copy()
            best_f_restart = best_f

            iters = 0
            max_iters = budget_per_restart

            for gen in range(max_iters):
                if time.time() > deadline:
                    break
                if iters >= budget_per_restart:
                    break

                # Generate population
                offspring = np.zeros((lam, dim))
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C_sqrt @ z)
                    offspring[i] = np.clip(y, -5, 5)
                    iters += 1
                    if time.time() > deadline:
                        break

                # Evaluate offspring
                fvals = np.full(lam, np.inf)
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    fvals[i] = func(offspring[i])

                # Sort by fitness
                idx = np.argsort(fvals)
                offspring = offspring[idx]
                fvals = fvals[idx]

                # Update best
                if fvals[0] < best_f_restart:
                    best_f_restart = fvals[0]
                    best_x_restart = offspring[0].copy()
                    if best_f_restart < best_f:
                        best_f = best_f_restart
                        best_x = best_x_restart.copy()

                # Compute weighted mean of selected
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * offspring[i]

                # Update step-size control vector
                d_sigma = 1.0 / damps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt.T @ ((x_new - x_mean) / sigma))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (iters + 1)) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(cs * (np.linalg.norm(ps) / chi_n - 1))

                # Update evolution path
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_new - x_mean) / sigma

                # Update covariance matrix
                delta = 0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (iters + 1)) <= (1.4 + 2 / (dim + 1)) * chi_n else 1
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (offspring[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + c1 * delta + c1 * (delta - 1) + cmu - c1) * C \
                    + c1 * (np.outer(pc, pc) + delta * np.eye(dim) - C) \
                    + cmu * rank_mu

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = (eigvecs * eigvals) @ eigvecs.T
                C = (C + C.T) / 2

                # Update Cholesky factor with periodic recomputation
                if gen % 5 == 0:
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        C_sqrt = np.eye(dim)

                x_mean = x_new
                iters += 1

            # After each restart, if no improvement, try a different starting point
            if best_f_restart >= best_f - 1e-10:
                # No improvement from this restart, try a new random start
                if time.time() < deadline and iters < budget - n_coarse:
                    new_start = rastrigin_aware_sample(1)[0]
                    new_f = func(new_start)
                    if new_f < best_f:
                        best_f = new_f
                        best_x = new_start.copy()

        # Phase 3: Final fine-tuning with small perturbations
        if time.time() < deadline - 5:
            calls_made = n_coarse + iters
            calls_left = budget - calls_made
            if calls_left > 0:
                for _ in range(min(calls_left, 20)):
                    if time.time() > deadline:
                        break
                    for scale in [0.1, 0.05, 0.01]:
                        x_try = best_x + scale * rng.standard_normal(dim)
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
