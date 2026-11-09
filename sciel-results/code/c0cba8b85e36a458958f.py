"""CMA-ES with adaptive restarts and Powell-based local refinement for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """CMA-ES with adaptive restarts and Gaussian Process refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget or check_time():
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        def cma_es(x0, sigma0, max_calls, rng):
            """
            Runs CMA-ES with per-evaluation stagnation tracking.
            Returns (best_x, best_f, calls_used, stagnated).
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

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            local_best_x = x.copy()
            local_best_f = np.inf
            calls_used = 0
            stagnation_counter = 0
            no_improve_threshold = 5 * lam

            # Initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_calls or check_time():
                    return local_best_x, local_best_f, calls_used, True
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                f, x_eval = call_func(x_trial)
                if f is None:
                    return local_best_x, local_best_f, calls_used, True
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            # Main loop
            while calls_used < max_calls and not check_time():
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                x_new = np.sum(weights[:, None] * pop[:mu], axis=0)

                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x_new - x)
                x = x_new

                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x_new - x)

                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2.0

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt @ z)
                    f, x_eval = call_func(x_trial)
                    if f is None:
                        return local_best_x, local_best_f, calls_used, True
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

                if stagnation_counter > no_improve_threshold:
                    return local_best_x, local_best_f, calls_used, True

                if sigma < 1e-10:
                    return local_best_x, local_best_f, calls_used, True

            return local_best_x, local_best_f, calls_used, True

        def powell_refine(x_center, f_center, max_iter):
            """
            Powell-based local refinement.
            Uses scipy.optimize.minimize with Powell method.
            Returns (best_x, best_f, calls_used).
            """
            nonlocal best_x, best_f, n_calls
            local_best_x = x_center.copy()
            local_best_f = f_center
            calls_used = 0

            # Wrapper to track calls and respect budget
            def wrapped_func(x):
                nonlocal n_calls, calls_used
                if n_calls >= budget or check_time():
                    return f_center # Return current best to stop search effectively? Or raise?
                # We need to be careful: scipy might call with arrays.
                x_clipped = np.clip(x, -5, 5)
                n_calls += 1
                calls_used += 1
                try:
                    f = float(func(x_clipped))
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_clipped.copy()
                        if f < best_f:
                            best_f = f
                            best_x = x_clipped.copy()
                    return f
                except Exception:
                    return f_center

            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x_center, 
                    method='Powell',
                    options={'maxiter': max_iter, 'disp': False}
                )
                # Update best from result if it's better
                if res.fun < local_best_f:
                    local_best_f = res.fun
                    local_best_x = res.x.copy()
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

            return local_best_x, local_best_f, calls_used

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling
        initial_candidates = [
            np.zeros(dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-4, 4, dim),
        ]

        for x in initial_candidates:
            f, x_eval = call_func(x)
            if f is not None:
                if f < best_f:
                    best_f = f
                    best_x = x_eval.copy()

        if n_calls >= budget or check_time():
            return best_x, best_f

        # Phase 2: CMA-ES with adaptive restarts
        current_x = best_x.copy()
        current_sigma = 1.5
        restart_count = 0
        max_restarts = 5

        while n_calls < budget and not check_time() and restart_count < max_restarts:
            remaining = budget - n_calls
            if remaining < 10:
                break

            if restart_count == max_restarts - 1:
                alloc = remaining
            else:
                alloc = int(remaining * 0.6)

            alloc = max(alloc, 20)

            res_x, res_f, used, stagnated = cma_es(current_x, current_sigma, alloc, rng)

            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

            restart_count += 1

            if not stagnated:
                current_x = res_x.copy()
                current_sigma = max(0.5, current_sigma * 0.7)
            else:
                # Perturb for next restart
                current_x = best_x + rng.uniform(-1.0, 1.0, dim)
                current_sigma = 1.5

        # Phase 3: Powell-based local refinement with remaining budget
        if n_calls < budget and not check_time():
            remaining = budget - n_calls
            if remaining >= 20:
                # Powell can use more evals, let's give it up to 100 or remaining
                max_iter = min(remaining, 100)
                _, _, _ = powell_refine(best_x, best_f, max_iter)

        return best_x, best_f
# EVOLVE-BLOCK-END
