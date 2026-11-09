"""Hybrid CMA-ES with (1+1)-ES polishing for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts and (1+1)-ES local polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

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
            Runs CMA-ES.
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

        def es_11_refine(x_center, f_center, max_calls, rng):
            """
            (1+1)-ES local refinement.
            Returns (best_x, best_f, calls_used).
            """
            nonlocal best_x, best_f, n_calls
            x = x_center.copy()
            sigma = 0.1  # Start with a small step for polishing
            local_best_x = x.copy()
            local_best_f = f_center
            calls_used = 0

            while calls_used < max_calls and not check_time():
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                f, x_eval = call_func(x_trial)
                if f is None:
                    break
                calls_used += 1

                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()
                    if f < best_f:
                        best_f = f
                        best_x = x_eval.copy()
                    x = x_eval.copy()
                    sigma *= 1.05  # Increase step on success
                else:
                    sigma *= 0.95  # Decrease step on failure

                # Stop if step becomes too small
                if sigma < 1e-6:
                    break

            return local_best_x, local_best_f, calls_used

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling
        # We use a mix of center and random points to ensure we have a good starting point.
        initial_candidates = [
            np.zeros(dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-4, 4, dim),
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
        # We allocate the bulk of the budget to CMA-ES.
        # We use a smaller number of restarts but allow each run to go longer if it's making progress.
        current_x = best_x.copy()
        current_sigma = 1.5
        restart_count = 0
        max_restarts = 3

        while n_calls < budget and not check_time() and restart_count < max_restarts:
            remaining = budget - n_calls
            if remaining < 10:
                break

            # Allocate a significant chunk to CMA-ES
            # If it's the last restart, use everything.
            if restart_count == max_restarts - 1:
                alloc = remaining
            else:
                # Use 70% of remaining budget for this CMA-ES run
                alloc = int(remaining * 0.7)

            alloc = max(alloc, 20)

            res_x, res_f, used, stagnated = cma_es(current_x, current_sigma, alloc, rng)

            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

            restart_count += 1

            if not stagnated:
                # If it didn't stagnate, we might have converged.
                # Restart with a smaller sigma around the best point found.
                current_x = res_x.copy()
                current_sigma = max(0.5, current_sigma * 0.5)
            else:
                # If it stagnated, perturb the center for the next restart.
                current_x = best_x + rng.uniform(-1.0, 1.0, dim)
                current_sigma = 1.5

        # Phase 3: (1+1)-ES local refinement with remaining budget
        # This is the structural change from the inspiration: replacing Powell with (1+1)-ES.
        if n_calls < budget and not check_time():
            remaining = budget - n_calls
            if remaining >= 10:
                es_11_refine(best_x, best_f, remaining, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
