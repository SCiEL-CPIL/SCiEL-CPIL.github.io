"""CMA-ES with adaptive restarts for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        def cma_es_single(x0, sigma0, max_calls, rng):
            """
            Run CMA-ES with adaptive restarts.
            Returns (x_best, f_best) for this run, updates global best.
            """
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            # Initialize state
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            x_mean = x0.copy()

            calls_used = 0
            generation = 0
            stagnation_count = 0
            best_f_run = np.inf
            best_x_run = x0.copy()

            # Initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            valid_idx = 0
            while valid_idx < lam and calls_used < max_calls and not check_time():
                z = rng.standard_normal(n)
                try:
                    L = np.linalg.cholesky(C)
                    x_trial = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    x_trial = x_mean + sigma * z

                f_val, x_eval = call_func(x_trial)
                if f_val is None:
                    break
                calls_used += 1
                pop[valid_idx] = x_eval
                fvals[valid_idx] = f_val
                if f_val < best_f_run:
                    best_f_run = f_val
                    best_x_run = x_eval.copy()
                valid_idx += 1

            if valid_idx < lam:
                return best_x_run, best_f_run

            # Main loop
            while calls_used < max_calls and not check_time():
                generation += 1

                # Sort by fitness
                order = np.argsort(fvals[:valid_idx])
                pop = pop[order]
                fvals = fvals[order]

                # Update mean
                x_mean_new = np.zeros(n)
                for i in range(mu):
                    x_mean_new += weights[i] * pop[i]

                # Update ps and pc
                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L).T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                y_mean = (x_mean_new - x_mean) / sigma
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * C_inv_sqrt @ y_mean

                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (generation + 1)))) < (1.4 + 2.0 / (n + 1)) * chi_n

                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_mean

                # Update covariance
                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                y_i = np.zeros((mu, n))
                for i in range(mu):
                    y_i[i] = (pop[i] - x_mean) / sigma
                C += cmu * (weights @ (y_i @ y_i.T))

                C = (C + C.T) / 2.0

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                x_mean = x_mean_new

                # Check convergence and stagnation
                if fvals[0] < best_f_run - 1e-12:
                    best_f_run = fvals[0]
                    best_x_run = pop[0].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Adaptive restart: increase sigma if stagnating
                if stagnation_count > 5:
                    sigma *= 1.5
                    stagnation_count = 0

                if sigma < 1e-10:
                    break

                # Generate new population
                pop = np.zeros((lam, n))
                fvals = np.zeros(lam)
                valid_idx = 0
                while valid_idx < lam and calls_used < max_calls and not check_time():
                    z = rng.standard_normal(n)
                    try:
                        L = np.linalg.cholesky(C)
                        x_trial = x_mean + sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        x_trial = x_mean + sigma * z

                    f_val, x_eval = call_func(x_trial)
                    if f_val is None:
                        break
                    calls_used += 1
                    pop[valid_idx] = x_eval
                    fvals[valid_idx] = f_val
                    valid_idx += 1

                if valid_idx < lam:
                    break

            return best_x_run, best_f_run

        rng = np.random.default_rng(42)

        # Phase 1: Initial random sampling
        n_init = min(10, budget // 10)
        for _ in range(n_init):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            call_func(x)

        if best_x is None:
            x = np.zeros(dim)
            call_func(x)
            if best_x is None:
                return None

        # Phase 2: CMA-ES with adaptive restarts
        # Run for remaining budget
        remaining = budget - n_calls
        if remaining > 0 and not check_time():
            x0 = best_x.copy()
            sigma0 = 2.0
            x_run, f_run = cma_es_single(x0, sigma0, remaining, rng)
            # Global best is already updated by call_func, but ensure consistency
            if f_run < best_f:
                best_f = f_run
                best_x = x_run

        return best_x, best_f
# EVOLVE-BLOCK-END
