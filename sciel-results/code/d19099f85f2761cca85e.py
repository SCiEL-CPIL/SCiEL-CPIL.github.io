"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with periodicity-aware restarts for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # population size
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = int(1 / np.sum(w ** 2))
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cma_run(x0, sigma0, remaining_budget):
            """Run CMA-ES from x0 with sigma0 for remaining_budget evaluations."""
            nonlocal deadline
            n_evals = 0
            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            best_x = x0.copy()
            best_f = func(x0)
            n_evals += 1

            for gen in range(remaining_budget // lam + 1):
                if n_evals >= remaining_budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                W = np.linalg.cholesky(C)
                ys = []
                fs = []
                for i in range(lam):
                    if n_evals >= remaining_budget:
                        break
                    if time.time() > deadline:
                        break
                    y = x + sigma * C_sqrt @ rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fs.append(fy)
                    n_evals += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(ys) < 2:
                    break

                ys = np.array(ys)
                fs = np.array(fs)
                # Sort by fitness
                idx = np.argsort(fs)
                ys = ys[idx]
                fs = fs[idx]

                # Update mean
                y_w = np.sum(w[:, None] * ys[:mu], axis=0)
                x = x + y_w

                # Compute covariance update
                y_sigma = np.sum(w[:, None] * (ys[:mu] - x + y_w), axis=0) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * damps) * y_sigma
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) <
                        (1.4 + 2 / (dim + 1)) * chi_n)
                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * damps) * y_sigma

                # Update C
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                    c1 * (np.outer(p_c, p_c) + delta_hsig * C) + \
                    cmu / sigma ** 2 * np.sum(w[:, None, None] * np.outer(ys[:mu] - x, ys[:mu] - x), axis=0)

                # Symmetrize
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C_sqrt (use Cholesky, fallback to eigen if fails)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Add small regularization
                    C += 1e-6 * np.eye(dim)
                    C = (C + C.T) / 2
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C_sqrt = np.eye(dim) * np.sqrt(np.trace(C) / dim)

            return best_x, best_f, n_evals

        rng = np.random.default_rng(42)

        # Multi-restart strategy: start with a few diverse initializations
        # and use CMA-ES with restarts when stagnation is detected
        best_x = None
        best_f = np.inf
        total_evals = 0

        # Initial diverse starts
        n_starts = 3
        starts = []
        for i in range(n_starts):
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = rng.uniform(-3, 3, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            starts.append(x0)

        # Run CMA-ES with restarts
        sigma_base = 2.0
        stagnation_threshold = 15 * lam
        min_improvement = 1e-6

        for start_idx, x0 in enumerate(starts):
            if total_evals >= budget:
                break
            if time.time() > deadline:
                break

            remaining = budget - total_evals
            x_best, f_best, n_used = cma_run(x0, sigma_base, remaining)
            total_evals += n_used

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

            # Check for stagnation - if improvement is minimal, try larger sigma restart
            if start_idx < n_starts - 1:
                # Try a restart centered on best with larger sigma
                if total_evals < budget and time.time() < deadline:
                    x_restart = best_x + rng.standard_normal(dim) * 0.5
                    x_restart = np.clip(x_restart, -5, 5)
                    remaining = budget - total_evals
                    x_best2, f_best2, n_used2 = cma_run(x_restart, sigma_base * 1.5, remaining)
                    total_evals += n_used2
                    if f_best2 < best_f:
                        best_f = f_best2
                        best_x = x_best2.copy()

        # If we still have budget, do a fine-tuning local search
        if total_evals < budget and best_x is not None and time.time() < deadline:
            # Simple local refinement
            sigma_local = 0.1
            x = best_x.copy()
            fx = best_f
            for _ in range(budget - total_evals):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_evals += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                else:
                    sigma_local *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
