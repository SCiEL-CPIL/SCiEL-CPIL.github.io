"""Persistent CMA-ES with adaptive kicks + (1+1) polish for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive kicks (restart-with-memory) and (1+1) polish."""

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
                x_clipped = np.clip(x, -5.0, 5.0)
                f = float(func(x_clipped))
                if not np.isfinite(f):
                    return None, None
                return f, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---------------- Persistent CMA-ES with adaptive kicks ----------------
        def cma_es(x0, sigma0, max_calls, rng):
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

            local_best_x = x0.copy()
            local_best_f = np.inf
            calls_used = 0
            gen_no_improve = 0
            max_gen_no_improve = 10

            def Cinv_sqrt():
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    return np.eye(n)

            def gen_pop():
                nonlocal calls_used, local_best_f, local_best_x
                nonlocal gen_no_improve
                pop = np.zeros((lam, n))
                fvals = np.zeros(lam)
                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt() @ z)
                    f, x_eval = call_func(x_trial)
                    if f is None:
                        break
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()
                        gen_no_improve = 0
                # pad any unfilled slots with the best available
                for i in range(lam):
                    if fvals[i] == 0.0 and np.all(pop[i] == 0.0):
                        j = np.argmin(fvals[:i]) if i > 0 else 0
                        if i > 0 and fvals[j] != 0.0:
                            pop[i] = pop[j]
                            fvals[i] = fvals[j]
                        else:
                            pop[i] = x
                            fvals[i] = local_best_f
                return pop, fvals

            # Initial population
            pop, fvals = gen_pop()
            if calls_used >= max_calls or check_time():
                return local_best_x, local_best_f, calls_used

            while calls_used < max_calls and not check_time():
                # Sort
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Weighted mean
                x_new = np.sum(weights[:, None] * pop[:mu], axis=0)

                Cinv_sqrt = Cinv_sqrt()

                # ps update
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (Cinv_sqrt @ (x_new - x))
                x = x_new

                # pc update
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) \
                    < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x_new - x)

                # Covariance update
                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2.0

                # Sigma update
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                # Track progress over this generation
                gen_best = local_best_f
                pop, fvals = gen_pop()
                if local_best_f >= gen_best:
                    gen_no_improve += 1
                else:
                    gen_no_improve = 0

                # Stagnation: inject a kick but KEEP the covariance (restart-with-memory)
                if gen_no_improve > max_gen_no_improve:
                    gen_no_improve = 0
                    # kick the mean toward a fresh random point
                    x = x + sigma * 5.0 * rng.standard_normal(n)
                    x = np.clip(x, -4.5, 4.5)
                    # enlarge the step to escape the basin
                    sigma *= 3.0
                    sigma = np.clip(sigma, 1e-15, 1e6)
                    # reset history vectors so the kick is not immediately smoothed away
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    # recenter covariance around the new mean via a fresh population
                    pop, fvals = gen_pop()
                    if calls_used >= max_calls or check_time():
                        break
                    continue

                # Convergence
                if sigma < 1e-10:
                    break

            return local_best_x, local_best_f, calls_used

        rng = np.random.default_rng(42)

        # ---------------- Phase 1: Diverse initial sampling ----------------
        init_points = [np.zeros(dim)]
        spreads = [0.5, 1.0, 2.0, 3.0, 4.0]
        for spread in spreads:
            for _ in range(2):
                init_points.append(rng.uniform(-spread, spread, dim))
        for _ in range(4):
            init_points.append(rng.uniform(-5.0, 5.0, dim))

        for x in init_points:
            if n_calls >= budget or check_time():
                break
            f, x_eval = call_func(x)
            if f is not None:
                update_best(x_eval, f)

        if best_f == np.inf:
            # no successful evaluation; seed at origin
            best_x = np.zeros(dim)
            best_f = 0.0

        # ---------------- Phase 2: Persistent CMA-ES with adaptive kicks ----------------
        if n_calls < budget and not check_time():
            # Reserve a small polish budget
            polish_budget = max(20, int(budget * 0.05))
            cma_budget = (budget - n_calls) - polish_budget
            if cma_budget < 30:
                cma_budget = budget - n_calls
                polish_budget = 0

            res_x, res_f, used = cma_es(best_x.copy(), 1.5, cma_budget, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        # ---------------- Phase 3: (1+1)-ES local polish ----------------
        if n_calls < budget and not check_time():
            local_sigma = max(0.05, 0.02 * np.sqrt(dim))
            base_sigma = local_sigma
            stall = 0
            while n_calls < budget and not check_time():
                x_trial = best_x + local_sigma * rng.standard_normal(dim)
                f, x_eval = call_func(x_trial)
                if f is None:
                    break
                if f < best_f:
                    best_f = f
                    best_x = x_eval.copy()
                    local_sigma = np.clip(local_sigma * 1.1, 1e-6, 5.0)
                    stall = 0
                else:
                    local_sigma *= 0.9
                    stall += 1
                    if stall > 20:
                        # occasionally re-open exploration a little
                        local_sigma = np.clip(base_sigma * 0.5, 1e-6, 5.0)
                        stall = 0
                        if local_sigma < 1e-4:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
