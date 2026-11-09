"""CMA-ES optimizer for Rastrigin rotated with periodicity-aware restarts."""
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES optimizer for Rastrigin rotated in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Efficient random initialization
        n_init = 10
        best_x = None
        best_f = np.inf

        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            n_init = 1

        calls_used = n_init

        # CMA-ES Parameters
        lam = 12  # Larger population for better exploration
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Periodicity-aware restart mechanism for Rastrigin
        rastrigin_period = 2.0 * math.sqrt(2.0)
        stagnation_threshold = 0.005  # More sensitive stagnation detection
        max_stagnant_gens = 10  # Restart sooner

        def cma_es_run(start_x, start_sigma, max_calls, time_limit):
            """Run CMA-ES from a given starting point."""
            nonlocal calls_used, best_f, best_x, rng

            mean = start_x.copy()
            sigma = start_sigma
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            local_calls = 0
            stagnant_gens = 0
            last_best_f = best_f

            while local_calls < max_calls and calls_used < budget:
                if time.time() > time_limit:
                    break

                # Generate population
                pop = np.zeros((lam, dim))
                try:
                    L = np.linalg.cholesky(C)
                except:
                    C = (C + C.T) / 2
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        break

                for i in range(lam):
                    if time.time() > time_limit:
                        break
                    z = rng.standard_normal(dim)
                    pop[i] = mean + sigma * (L @ z)

                # Evaluate population
                fvals = np.zeros(lam)
                for i in range(lam):
                    if calls_used >= budget:
                        break
                    if time.time() > time_limit:
                        break
                    fvals[i] = func(pop[i])
                    calls_used += 1
                    local_calls += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()

                if calls_used >= budget:
                    break

                # Check for stagnation
                if best_f < last_best_f - stagnation_threshold:
                    stagnant_gens = 0
                    last_best_f = best_f
                else:
                    stagnant_gens += 1

                # Periodicity-aware restart: if stagnant, shift mean by half-period
                if stagnant_gens >= max_stagnant_gens and calls_used < budget - lam:
                    # Shift along a random direction by approximately half the Rastrigin period
                    shift_dir = rng.standard_normal(dim)
                    shift_dir /= np.linalg.norm(shift_dir)
                    shift_mag = rastrigin_period / 2.0
                    mean = best_x + shift_mag * shift_dir
                    # Clip to box
                    mean = np.clip(mean, -4.5, 4.5)
                    sigma = 2.0  # Larger sigma for broader exploration after restart
                    C = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    stagnant_gens = 0
                    last_best_f = best_f
                    continue

                # Sort
                order = np.argsort(fvals)
                y = np.zeros((mu, dim))
                for i in range(mu):
                    y[i] = pop[order[i]]

                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)

                # Update paths
                v = (new_mean - mean) / sigma

                # Compute C^{-1/2}
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                ps = (1 - c_sigma) * ps + c_sigma / math.sqrt(c_sigma) * C_inv_sqrt @ v

                norm_ps = np.linalg.norm(ps)
                t_gen = max(local_calls, 1) / lam
                h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma)**(2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)

                pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v

                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                delta = (1 + delta_h) * (1 - c_1)

                rank1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_norm = y[i] - mean
                    rank_mu += weights[i] * np.outer(y_norm, y_norm)

                C = delta * C + c_1 * (rank1 - C) + c_mu * rank_mu / (sigma**2)
                C = (C + C.T) / 2

                # Update sigma
                sigma = sigma * math.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))

                mean = new_mean

            return best_x, best_f

        # Run CMA-ES with periodicity-aware restarts - larger initial sigma
        remaining_budget = budget - calls_used
        cma_es_run(best_x, 2.0, remaining_budget, deadline)

        return (best_x, best_f)
