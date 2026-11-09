"""CMA-ES with adaptive restarts and local polish for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and local (1+1)-ES polish for multimodal Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

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
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        def update(x, f):
            nonlocal best_x, best_f
            if f is not None and f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(x0, sigma0, max_calls, rng):
            """Run CMA-ES with strict call tracking."""
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.0) - np.log(np.arange(1, mu + 1) + 0.0)
            weights = np.where(np.isfinite(weights), weights, 0.0)
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0

            x = x0.copy()
            calls_used = 0
            local_best_x = x.copy()
            local_best_f = np.inf

            # Generate initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if check_time() or calls_used >= max_calls:
                    break
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                f, x_eval = call_func(x_trial)
                if f is None:
                    return local_best_x, local_best_f, calls_used
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()

            # Main CMA-ES loop
            while calls_used < max_calls and not check_time():
                # Sort by fitness
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update mean
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * pop[i]

                # Compute Cinv_sqrt for ps update
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                # Update ps and pc
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x - x_new)
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x - x_new)

                # Update covariance
                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2.0

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                x = x_new

                # Generate new population
                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt @ z)
                    f, x_eval = call_func(x_trial)
                    if f is None:
                        return local_best_x, local_best_f, calls_used
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()

                # Convergence check
                if sigma < 1e-12:
                    break

            return local_best_x, local_best_f, calls_used

        def local_polish(x0, max_calls, rng):
            """(1+1)-ES local search to refine the solution."""
            n = dim
            x = x0.copy()
            f, _ = call_func(x)
            if f is None:
                return x, np.inf, 0
            calls_used = 1
            sigma = 0.5
            local_best_x = x.copy()
            local_best_f = f

            while calls_used < max_calls and not check_time():
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy, _ = call_func(y)
                if fy is None:
                    break
                calls_used += 1
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.1
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break
            return local_best_x, local_best_f, calls_used

        rng = np.random.default_rng(42)

        # Phase 1: Initial random sampling
        n_initial = min(8, budget // 50)
        for i in range(n_initial):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f, x_eval = call_func(x)
            if f is None:
                break
            update(x_eval, f)

        if best_x is None:
            x = np.zeros(dim)
            f, x_eval = call_func(x)
            if f is not None:
                update(x_eval, f)
            if best_x is None:
                return None, np.inf

        # Phase 2: CMA-ES restarts
        remaining = budget - n_calls
        if remaining < 20:
            return best_x, best_f

        # Allocate budget: 60% to first restart, 20% to second, 20% to third
        n_restarts = 3
        restart_budgets = [int(remaining * 0.55), int(remaining * 0.25), int(remaining * 0.20)]
        # Ensure minimum budget per restart
        for i in range(n_restarts):
            restart_budgets[i] = max(restart_budgets[i], 10)

        starts = [best_x.copy()]
        for _ in range(n_restarts - 1):
            starts.append(rng.uniform(-5, 5, dim))

        for i, x0 in enumerate(starts):
            if check_time() or n_calls >= budget:
                break
            max_calls = min(restart_budgets[i], budget - n_calls)
            if max_calls < 10:
                break

            sigma0 = 2.0 if i == 0 else 1.0
            res_x, res_f, used = cma_es(x0, sigma0, max_calls, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        # Phase 3: Local polish on the best solution
        remaining = budget - n_calls
        if remaining > 10 and best_x is not None:
            polish_calls = min(remaining, 50)
            pol_x, pol_f, used = local_polish(best_x, polish_calls, rng)
            if pol_f < best_f:
                best_f = pol_f
                best_x = pol_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
