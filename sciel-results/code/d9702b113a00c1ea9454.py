"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and deadline guard for 5D box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 330s
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # State tracking
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        calls_used = 0

        def check_deadline():
            nonlocal calls_used
            if time.time() > deadline:
                return True
            if calls_used >= budget:
                return True
            return False

        def safe_call(x):
            nonlocal calls_used, best_x, best_f
            if check_deadline():
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~9 for dim=5
        pop_size = min(pop_size, budget // 5)  # ensure we can do multiple generations

        # Phase 1: Initial sampling to find a good starting region
        n_init = min(20, budget // 10)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is None:
                break

        if np.isnan(best_x).any():
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: CMA-ES with restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if check_deadline():
                break

            # Initialize CMA-ES around current best
            x_mean = best_x.copy()
            if restart == 0:
                sigma = 1.5
            else:
                sigma = 0.5 * (0.8 ** restart)

            # CMA-ES state
            C = np.eye(dim)
            C_inv = np.eye(dim)
            step_size = 1.0
            # Precompute
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            gen = 0
            no_improve = 0
            best_f_this_restart = best_f

            while not check_deadline():
                # Generate population
                ys = []
                fs = []
                for i in range(pop_size):
                    if check_deadline():
                        break
                    z = rng.standard_normal(dim)
                    # Use Cholesky of C for sampling
                    try:
                        C_chol = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C_chol = np.eye(dim)
                    y = x_mean + sigma * (C_chol @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    ys.append(y)
                    fs.append(fy)

                if len(ys) < 2:
                    break

                ys = np.array(ys)
                fs = np.array(fs)

                # Sort by fitness
                order = np.argsort(fs)
                ys = ys[order]
                fs = fs[order]

                # Update mean
                x_new_mean = np.sum(weights[:, None] * ys[:mu], axis=0)

                # Update covariance matrix
                y_w = (ys[:mu] - x_mean) / sigma
                delta_sigma = (np.linalg.norm(x_new_mean - x_mean) / (c_sigma * chi_n * sigma) - 1)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_sigma * (x_new_mean - x_mean) / (sigma + 1e-30)

                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) / chi_n < (1.4 + 2 / (dim + 1))
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma

                C = ((1 - c_1 - c_mu) * C
                     + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C - np.eye(dim))
                     + c_mu * np.sum(weights[:, None, None] * np.outer(y_w, y_w), axis=0))

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    C_inv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv = np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 50.0)

                x_mean = x_new_mean

                # Check for improvement
                if fs[0] < best_f_this_restart:
                    best_f_this_restart = fs[0]
                    no_improve = 0
                else:
                    no_improve += 1

                gen += 1

                # Restart if stuck
                if no_improve > 3 * dim:
                    break

                # Check convergence
                if np.linalg.norm(sigma * np.sqrt(np.diag(C))) < 1e-10:
                    break

            # After CMA-ES, do local refinement around best
            if not check_deadline():
                local_sigma = 0.3
                for _ in range(min(50, budget - calls_used)):
                    if check_deadline():
                        break
                    y = best_x + local_sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        local_sigma *= 1.3
                    else:
                        local_sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
