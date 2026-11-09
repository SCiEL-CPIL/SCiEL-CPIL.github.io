import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 300s limit, 5s safety margin

        dim = self.dim
        budget = self.budget

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        calls_used = 0

        def update_global(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()

        # CMA-ES function for a single run
        def run_cmaes(x_init, sigma_init, max_calls, deadline_t):
            nonlocal calls_used, global_best_x, global_best_f

            # CMA-ES Parameters
            lam = 10
            mu = lam // 2

            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights**2)

            # Initial parameters
            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)

            # Adaptation parameters
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 - mueff / (dim + 4)) / (dim + 4 + mueff)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

            # Path variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim**2))
            c_hsig = (1 + 2 / max(1, mueff - 1) + c_sigma) / (1 + c_sigma)

            # Stagnation tracking
            best_f_run = np.inf
            stagnation_count = 0
            max_stagnation = 15  # restart after 15 generations without improvement

            gen = 0

            while calls_used + lam <= max_calls and time.time() < deadline_t:
                gen += 1

                # Compute Cholesky with regularization
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except np.linalg.LinAlgError:
                    try:
                        L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)

                # Generate population
                z = np.random.randn(lam, dim)
                y = z @ L.T
                x_pop = x_mean + sigma * y
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate
                f_pop = np.array([func(x) for x in x_pop])
                calls_used += lam

                # Update global best
                idx_best = np.argmin(f_pop)
                if f_pop[idx_best] < global_best_f:
                    global_best_f = f_pop[idx_best]
                    global_best_x = x_pop[idx_best].copy()

                # Stagnation check
                if f_pop[idx_best] < best_f_run - 1e-10:
                    best_f_run = f_pop[idx_best]
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Restart if stagnated
                if stagnation_count >= max_stagnation and gen > 5:
                    break

                # Sort
                idx_sort = np.argsort(f_pop)
                x_sorted = x_pop[idx_sort]
                y_sorted = y[idx_sort]

                # Selection
                x_w = x_sorted[:mu]
                y_w = y_sorted[:mu]

                # Update mean
                x_mean_old = x_mean
                x_mean = np.sum(weights[:, np.newaxis] * x_w, axis=0)

                # Update p_sigma
                d_mean = x_mean - x_mean_old
                d_mean_norm = d_mean / sigma

                # Solve L^T v = d_mean_norm (more stable than inv)
                try:
                    L_inv_T_dmean = np.linalg.solve(L.T, d_mean_norm)
                except np.linalg.LinAlgError:
                    L_inv_T_dmean = d_mean_norm

                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * L_inv_T_dmean

                # h_sigma
                if gen > 0:
                    norm_ps = np.linalg.norm(p_sigma)
                    denom = np.sqrt(max(1e-16, 1 - (1 - c_sigma) ** (2 / gen)))
                    h_sigma = 1.0 if (norm_ps / denom) < (1.4 + 2 / (dim + 1)) * c_hsig else 0.0
                else:
                    h_sigma = 1.0

                # Update sigma
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
                sigma = np.clip(sigma, 1e-8, 20.0)

                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * L_inv_T_dmean

                # Update C
                y_w_centered = y_w - np.mean(y_w, axis=0)
                rank_mu = np.sum(weights[:, np.newaxis]**2 * y_w_centered[:, np.newaxis] * y_w_centered, axis=0)
                C = (1 - c_1 - c_mu) * C \
                    + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                    + c_mu * rank_mu

                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2

                # Periodic regularization to prevent numerical issues
                if gen % 20 == 0:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                        C = C + (1e-10 - np.min(eigvals)) * np.eye(dim)

            return stagnation_count >= max_stagnation and gen > 5

        # Phase 1: Initial CMA-ES run
        x_start = np.zeros(dim)
        sigma_start = 2.5
        max_calls_phase1 = int(budget * 0.6)
        restart_needed = run_cmaes(x_start, sigma_start, max_calls_phase1, deadline)

        # Phase 2: If restart needed or budget remains, do a second run from best point
        if global_best_x is not None and calls_used < budget and time.time() < deadline:
            remaining = budget - calls_used
            if remaining >= lam:
                # Second run: smaller sigma, centered on best
                sigma_phase2 = max(0.5, 0.2 * 2.5)
                x_phase2 = global_best_x.copy()
                run_cmaes(x_phase2, sigma_phase2, budget, deadline)

        # Phase 3: Local polish using coordinate descent with adaptive step
        if global_best_x is not None and calls_used < budget and time.time() < deadline:
            polish_deadline = deadline - 2.0
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            for step in step_sizes:
                if calls_used >= budget or time.time() > polish_deadline:
                    break
                improved = False
                for _ in range(3):  # 3 passes per step size
                    if calls_used >= budget or time.time() > polish_deadline:
                        break
                    for i in range(dim):
                        if calls_used >= budget or time.time() > polish_deadline:
                            break
                        for direction in [1, -1]:
                            if calls_used >= budget or time.time() > polish_deadline:
                                break
                            candidate = global_best_x.copy()
                            candidate[i] += direction * step
                            candidate = np.clip(candidate, -5, 5)
                            f_val = func(candidate)
                            calls_used += 1
                            if f_val < global_best_f:
                                global_best_f = f_val
                                global_best_x = candidate
                                improved = True
                if not improved:
                    break

        # Final: a few random perturbations around best to escape tiny local minima
        if global_best_x is not None and calls_used < budget and time.time() < deadline:
            polish_deadline = deadline - 1.0
            for _ in range(10):
                if calls_used >= budget or time.time() > polish_deadline:
                    break
                perturbation = np.random.randn(dim) * 0.01
                candidate = np.clip(global_best_x + perturbation, -5, 5)
                f_val = func(candidate)
                calls_used += 1
                if f_val < global_best_f:
                    global_best_f = f_val
                    global_best_x = candidate

        return global_best_x, global_best_f
