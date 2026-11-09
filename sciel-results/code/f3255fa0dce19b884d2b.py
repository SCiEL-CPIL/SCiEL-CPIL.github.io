"""Multi-island CMA-ES with GP-based local refinement for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-island CMA-ES with GP-Expected Improvement refinement for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

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
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        def cma_es_island(x0, sigma0, max_calls, rng):
            """Run one CMA-ES island. Returns (x_best, f_best) for this island."""
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

            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            x_mean = x0.copy()

            calls_used = 0
            generation = 0
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

            while calls_used < max_calls and not check_time():
                generation += 1
                order = np.argsort(fvals[:valid_idx])
                pop = pop[order]
                fvals = fvals[order]

                x_mean_new = np.zeros(n)
                for i in range(mu):
                    x_mean_new += weights[i] * pop[i]

                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L).T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                y_mean = (x_mean_new - x_mean) / sigma
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (C_inv_sqrt @ y_mean)

                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (generation + 1)))) < (1.4 + 2.0 / (n + 1)) * chi_n

                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_mean

                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                y_i = np.zeros((mu, n))
                for i in range(mu):
                    y_i[i] = (pop[i] - x_mean) / sigma
                C += cmu * (weights @ (y_i @ y_i.T))
                C = (C + C.T) / 2.0

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)
                x_mean = x_mean_new

                if fvals[0] < best_f_run - 1e-12:
                    best_f_run = fvals[0]
                    best_x_run = pop[0].copy()

                if sigma < 1e-10:
                    break

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

        def gp_refine(x_best, f_best, max_calls, rng):
            """Gaussian Process with Expected Improvement local refinement."""
            n = dim
            x_best = np.array(x_best, dtype=float)
            f_best = float(f_best)

            # Collect initial training points around x_best
            X_train = [x_best.copy()]
            Y_train = [f_best]

            # Add a few local samples to build initial GP
            n_local = min(10, max_calls)
            for _ in range(n_local):
                if check_time() or n_calls >= budget:
                    break
                x_trial = x_best + rng.normal(0, 0.3, n)
                x_trial = np.clip(x_trial, -5, 5)
                f_val, x_eval = call_func(x_trial)
                if f_val is not None:
                    X_train.append(x_eval)
                    Y_train.append(f_val)
                    if f_val < f_best:
                        f_best = f_val
                        x_best = x_eval.copy()

            X_train = np.array(X_train)
            Y_train = np.array(Y_train)

            # GP kernel: Matern 5/2
            def gp_predict(X_pred, X_tr, Y_tr, length_scale=1.0, sigma_f=1.0, noise=1e-8):
                n_tr = len(X_tr)
                n_pred = X_pred.shape[0]

                # Compute kernel matrix
                def matern52(dist):
                    sqrt3d = np.sqrt(3.0) * dist
                    return (1.0 + sqrt3d + (sqrt3d ** 2) / 3.0) * np.exp(-sqrt3d)

                D_tr = X_tr[:, np.newaxis, :] - X_tr[np.newaxis, :, :]
                dist_tr = np.sqrt(np.sum(D_tr ** 2, axis=-1))
                K = matern52(dist_tr / length_scale) * sigma_f ** 2 + np.eye(n_tr) * noise

                D_pred = X_pred[:, np.newaxis, :] - X_tr[np.newaxis, :, :]
                dist_pred = np.sqrt(np.sum(D_pred ** 2, axis=-1))
                K_pred = matern52(dist_pred / length_scale) * sigma_f ** 2

                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_tr))
                    mean = K_pred @ alpha
                    v = np.linalg.solve(L, K_pred.T)
                    cov = sigma_f ** 2 - v.T @ v
                    var = np.diag(cov).clip(min=1e-12)
                except np.linalg.LinAlgError:
                    # Fallback: simple distance-based prediction
                    dists = np.sqrt(np.sum((X_pred[:, np.newaxis, :] - X_tr[np.newaxis, :, :]) ** 2, axis=-1))
                    w = np.exp(-dists ** 2 / (2 * length_scale ** 2))
                    w /= w.sum(axis=1, keepdims=True) + 1e-12
                    mean = w @ Y_tr
                    var = np.full(n_pred, sigma_f ** 2)

                return mean, np.sqrt(var)

            # Expected Improvement acquisition
            from scipy.stats import norm

            def expected_improvement(x_pred, X_tr, Y_tr, f_current, length_scale, sigma_f):
                if len(X_tr) < 2:
                    return 0.0
                mu, sigma = gp_predict(x_pred.reshape(1, -1), X_tr, Y_tr, length_scale, sigma_f, 1e-8)
                mu = mu[0]
                sigma = sigma[0]
                if sigma < 1e-10:
                    return 0.0
                z = (f_current - mu) / sigma
                ei = (f_current - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return max(ei, 0.0)

            # Adaptive length scale
            length_scale = 0.5
            sigma_f = max(np.std(Y_train) if len(Y_train) > 1 else 1.0, 0.1)

            gp_calls = 0
            stagnation = 0
            best_gp_f = f_best
            best_gp_x = x_best.copy()

            while gp_calls < max_calls and not check_time() and n_calls < budget:
                # Generate candidate points
                n_candidates = 20
                candidates = x_best + rng.normal(0, length_scale, (n_candidates, n))
                candidates = np.clip(candidates, -5, 5)

                # Evaluate EI for each candidate
                eis = np.zeros(n_candidates)
                for i in range(n_candidates):
                    eis[i] = expected_improvement(candidates[i], X_train, Y_train, best_gp_f, length_scale, sigma_f)

                # Pick best candidate by EI
                idx = np.argmax(eis)
                x_trial = candidates[idx]
                f_val, x_eval = call_func(x_trial)
                if f_val is None:
                    break
                gp_calls += 1

                X_train = np.vstack([X_train, x_eval])
                Y_train = np.append(Y_train, f_val)

                if f_val < best_gp_f:
                    best_gp_f = f_val
                    best_gp_x = x_eval.copy()
                    stagnation = 0
                    # Shrink exploration
                    length_scale *= 0.85
                    sigma_f *= 0.9
                else:
                    stagnation += 1
                    if stagnation > 8:
                        # Reset exploration slightly
                        length_scale *= 1.2
                        stagnation = 0

                length_scale = max(length_scale, 0.05)
                sigma_f = max(sigma_f, 0.05)

            return best_gp_x, best_gp_f

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling
        n_init = 15
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

        remaining = budget - n_calls

        # Phase 2: Multi-island CMA-ES (60% of remaining)
        n_islands = 3
        budget_per_island = max(10, int(remaining * 0.6 / n_islands))
        island_budget = min(budget_per_island, remaining)

        # Start islands from best point and random diverse points
        island_starts = [best_x.copy()]
        for _ in range(n_islands - 1):
            if check_time() or n_calls >= budget:
                break
            x_init = best_x + rng.normal(0, 2.0, dim)
            x_init = np.clip(x_init, -5, 5)
            island_starts.append(x_init)

        for i, x_start in enumerate(island_starts):
            if check_time() or n_calls >= budget:
                break
            sigma0 = 1.5 if i == 0 else 2.5
            calls_for_island = min(island_budget, budget - n_calls)
            if calls_for_island < 4:
                break
            x_island, f_island = cma_es_island(x_start, sigma0, calls_for_island, rng)

        # Phase 3: GP-based local refinement (40% of remaining)
        gp_budget = max(10, budget - n_calls)
        if gp_budget > 5 and not check_time():
            x_refined, f_refined = gp_refine(best_x, best_f, gp_budget, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
