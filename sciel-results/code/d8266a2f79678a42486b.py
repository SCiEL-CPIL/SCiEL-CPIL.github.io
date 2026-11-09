"""CMA-ES with adaptive restarts and (1+1)-ES polishing for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES with adaptive restarts for the multimodal Weierstrass function (f16).
    Uses a population-based evolution strategy with adaptive step size and
    covariance matrix, combined with (1+1)-ES local polishing.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                val = float(func(x_clipped))
                return val, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                return True
            return False

        rng = np.random.default_rng(42)

        # --- (1+1)-ES Local Polisher ---
        def local_polish_11es(x0, f0, max_calls, sigma0=None):
            """
            Run (1+1)-ES from x0. Returns best_x, best_f, calls_used.
            """
            nonlocal n_calls, best_x, best_f

            calls_used = 0
            x_best = x0.copy()
            f_best = f0
            sigma = sigma0 if sigma0 is not None else 0.5

            while calls_used < max_calls and n_calls < budget and not check_time():
                # Generate perturbation
                direction = rng.standard_normal(dim)
                x_trial = x_best + sigma * direction
                x_trial = np.clip(x_trial, -5, 5)

                f_trial, x_trial_eval = call_func(x_trial)
                if f_trial is None:
                    break
                calls_used += 1

                if f_trial < f_best:
                    x_best = x_trial_eval
                    f_best = f_trial
                    sigma = max(sigma * 1.15, 0.01)  # Increase step on success
                else:
                    sigma = max(sigma * 0.85, 0.01)  # Decrease step on failure

                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()

            return x_best, f_best, calls_used

        # --- CMA-ES Implementation ---
        def cma_es(x0, sigma0, max_calls, pop_size=None):
            """
            Run CMA-ES. Returns best_x, best_f, calls_used.
            """
            nonlocal n_calls, best_x, best_f

            if pop_size is None:
                pop_size = max(4, int(4 + 3 * np.log(dim)))

            # CMA-ES parameters
            mu = pop_size // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs

            # Initialize
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # Population
            calls_used = 0
            x_best_local = x0.copy()
            f_best_local = np.inf

            # Evaluate initial point
            f0, x0_eval = call_func(x0)
            if f0 is not None:
                calls_used += 1
                f_best_local = f0
                x_best_local = x0_eval

            # Main loop
            while calls_used < max_calls and n_calls < budget and not check_time():
                # Generate population
                X = np.zeros((pop_size, dim))
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    X[i] = x_mean + sigma * (C_sqrt @ z)
                    X[i] = np.clip(X[i], -5, 5)

                # Evaluate population
                F = np.full(pop_size, np.inf)
                for i in range(pop_size):
                    f_i, x_i_eval = call_func(X[i])
                    if f_i is not None:
                        calls_used += 1
                        F[i] = f_i
                        if f_i < f_best_local:
                            f_best_local = f_i
                            x_best_local = x_i_eval.copy()

                # Sort by fitness
                idx = np.argsort(F)
                X_sorted = X[idx]
                F_sorted = F[idx]

                # Update mean
                x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                # Update covariance matrix
                diff = (x_new - x_mean) / sigma
                pc = (1 - cc) * pc + cc * diff
                norm_pc = np.linalg.norm(pc)
                hsig = (norm_pc / np.sqrt(1 - (1 - cc)**2)) < (1.4 + 2 / (dim + 1)) * chi_n

                ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * diff
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = (X_sorted[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)

                C = (1 - cs * c1 - cs * cmu) * C + cs * c1 * rank_one + cs * cmu * rank_mu
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))

                # Update sigma
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(max(sigma, 1e-8), 50.0)

                x_mean = x_new

            return x_best_local, f_best_local, calls_used

        # --- Phase 1: Initial Sampling ---
        n_initial = 10
        for i in range(n_initial):
            if n_calls >= budget or check_time():
                break
            x = rng.uniform(-5, 5, dim)
            f, x_eval = call_func(x)
            if f is None:
                continue
            update_best(x_eval, f)

        if best_x is None:
            return best_x, best_f

        # --- Phase 2: CMA-ES with Adaptive Restarts ---
        # Start CMA-ES from the best point found
        max_cma_calls = min(budget - n_calls, 300)
        if max_cma_calls > 10 and not check_time():
            x_cma, f_cma, calls_used = cma_es(best_x, 1.0, max_cma_calls)
            if f_cma < best_f:
                best_f = f_cma
                best_x = x_cma

        # --- Phase 3: (1+1)-ES Polish ---
        if best_x is not None and n_calls < budget and not check_time():
            remaining = budget - n_calls
            max_polish = min(remaining, 50)
            if max_polish > 5:
                x_polished, f_polished, calls_used = local_polish_11es(best_x, best_f, max_polish, sigma0=0.3)
                if f_polished < best_f:
                    best_x = x_polished
                    best_f = f_polished

        # --- Phase 4: If budget remains, do another round of CMA-ES from a different start ---
        if n_calls < budget and not check_time():
            remaining = budget - n_calls
            if remaining > 20:
                # Try from a random point near the best
                x_start = best_x + rng.normal(0, 0.5, dim)
                x_start = np.clip(x_start, -5, 5)
                max_cma_calls2 = min(remaining, 100)
                x_cma2, f_cma2, calls_used2 = cma_es(x_start, 0.5, max_cma_calls2)
                if f_cma2 < best_f:
                    best_f = f_cma2
                    best_x = x_cma2

                # Final polish
                if best_x is not None and n_calls < budget and not check_time():
                    remaining2 = budget - n_calls
                    max_polish2 = min(remaining2, 30)
                    if max_polish2 > 5:
                        x_polished2, f_polished2, calls_used2 = local_polish_11es(best_x, best_f, max_polish2, sigma0=0.2)
                        if f_polished2 < best_f:
                            best_x = x_polished2
                            best_f = f_polished2

        return best_x, best_f

# EVOLVE-BLOCK-END
