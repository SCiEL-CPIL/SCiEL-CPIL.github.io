"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using CMA-ES-like strategy with multi-restart and adaptive step size.
    Focuses on efficient exploration and exploitation for multimodal and unimodal functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Initial sampling to get a good starting point
        # Sample 10 points: center + 9 random
        for i in range(10):
            if calls[0] >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            try:
                f_val = wrapped(x)
                update_best(x, f_val)
            except Exception:
                pass

        # Phase 2: CMA-ES-like optimization with adaptive step size
        # Use a population-based strategy with covariance adaptation
        if best_x is not None and calls[0] < budget:
            # Population size: lambda = 4 + floor(3 * log(dim))
            lam = max(6, 4 + int(3 * np.log(dim)))
            # Step size (sigma)
            sigma = 2.0  # Start with a reasonably large step
            x_mean = best_x.copy()
            x_best = best_x.copy()
            f_best = best_f

            # Covariance matrix (start with identity)
            C = np.eye(dim)
            # Evolution path for covariance adaptation
            p_sigma = np.zeros(dim)
            # Evolution path for mean
            p_c = np.zeros(dim)

            # Adaptation parameters
            c_sigma = (lam / 2 + 1) / (dim + 2 + lam / 2)
            d_sigma = 1 + 2 * max(0, (lam / 2 - 1) / (dim + 1)) + c_sigma
            c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
            mu = lam // 2  # Number of selected individuals
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1 / np.sum(w ** 2)

            # Track best
            best_x_es = x_best.copy()
            best_f_es = f_best

            # Main CMA-ES loop
            while calls[0] < budget and time.time() < deadline:
                # Check time
                if time.time() > deadline:
                    break

                # Generate population
                pop = []
                pop_f = []

                for i in range(lam):
                    if calls[0] >= budget or time.time() > deadline:
                        break

                    # Sample from multivariate normal
                    # Use Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)

                    z = rng.standard_normal(dim)
                    x_new = x_mean + sigma * L @ z
                    x_new = np.clip(x_new, -5, 5)

                    try:
                        f_new = wrapped(x_new)
                    except Exception:
                        continue

                    pop.append(x_new)
                    pop_f.append(f_new)

                    if f_new < best_f_es:
                        best_x_es = x_new.copy()
                        best_f_es = f_new

                if len(pop) == 0:
                    break

                # Sort by fitness
                idx = np.argsort(pop_f)
                pop = [pop[i] for i in idx]
                pop_f = [pop_f[i] for i in idx]

                # Update best
                if pop_f[0] < best_f:
                    update_best(pop[0], pop_f[0])

                # Update mean
                x_new_mean = np.sum([w[i] * pop[i] for i in range(min(mu, len(pop)))], axis=0)

                # Update sigma (CSA)
                try:
                    C_inv_sqrt = np.linalg.inv(L)
                    x_old_mean_norm = np.linalg.norm(C_inv_sqrt @ (x_new_mean - x_mean))
                    x_new_mean_norm = np.linalg.norm(C_inv_sqrt @ (x_new_mean - x_mean))
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new_mean - x_mean)) / sigma
                    h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls[0] // lam + 1)) / np.sqrt(1 - (1 - c_sigma) ** 2) if np.linalg.norm(p_sigma) > 0 else 1
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) - 1))
                    sigma = np.clip(sigma, 1e-8, 10)
                except Exception:
                    pass

                # Update covariance
                try:
                    if h_sigma < 1.7 + 2 / (dim + 1):
                        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
                    else:
                        p_c = (1 - c_c) * p_c

                    rank_mu = np.outer(x_new_mean - x_mean, x_new_mean - x_mean)
                    for i in range(min(mu, len(pop))):
                        diff = pop[i] - x_mean
                        rank_mu += w[i] * np.outer(diff, diff)

                    C = (1 + c_c * (2 - c_c) - c_c / mu_eff) * C + c_c * np.outer(p_c, p_c) + c_c * rank_mu / sigma ** 2
                    # Ensure positive semi-definite
                    C = (C + C.T) / 2
                except Exception:
                    pass

                x_mean = x_new_mean
                x_mean = np.clip(x_mean, -5, 5)

                # If sigma is very small, we've converged
                if sigma < 1e-7:
                    break

                # If no improvement in last few generations, restart
                if calls[0] > 50 and time.time() < deadline:
                    # Check if we're making progress
                    if best_f_es == best_f and calls[0] > 100:
                        # Try a restart with perturbation
                        if rng.random() < 0.1:
                            x_mean = best_x + rng.standard_normal(dim) * 0.5
                            x_mean = np.clip(x_mean, -5, 5)
                            sigma = max(sigma, 0.5)
                            C = np.eye(dim)
                            p_sigma = np.zeros(dim)
                            p_c = np.zeros(dim)

        # Phase 3: Final polish with (1+1)-ES from the best point
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start with a moderate step
            no_improve = 0
            max_no_improve = 50

            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1

                if no_improve > max_no_improve:
                    break
                if sigma < 1e-8:
                    break

        if best_x is None:
            # Fallback
            best_x = rng.uniform(-5, 5, dim)
            try:
                best_f = wrapped(best_x)
            except Exception:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
