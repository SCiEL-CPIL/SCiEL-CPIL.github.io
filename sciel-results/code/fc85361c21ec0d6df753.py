"""CMA-ES + Gaussian Process surrogate for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """GP-guided CMA-ES hybrid for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        eval_count = 0
        all_x = []
        all_f = []

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            all_x.append(x.copy())
            all_f.append(f_val)
            return f_val

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Structured initial sampling ---
        # Use a mix of lattice and random points for good early coverage
        n_init = min(40, budget // 5)
        init_points = []
        # Lattice points
        for v1 in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            for v2 in [-4.0, 0.0, 4.0]:
                for v3 in [-4.0, 0.0, 4.0]:
                    for v4 in [-4.0, 0.0, 4.0]:
                        for v5 in [-4.0, 0.0, 4.0]:
                            init_points.append(np.array([v1, v2, v3, v4, v5]))
        rng.shuffle(init_points)
        # Add random points
        for _ in range(n_init):
            init_points.append(rng.uniform(-4.5, 4.5, dim))
        init_points = init_points[:n_init]

        for pt in init_points:
            if eval_count >= budget - 50 or time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # --- Phase 2: CMA-ES with GP-guided restarts ---
        def cmaes_run(start_x, sigma0, max_evals, max_time):
            nonlocal eval_count, deadline
            n = dim
            # Adaptive population size
            lambda_ = int(4 + 3 * n)
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            mean = start_x.copy()
            sigma = sigma0
            C = np.eye(n)
            ps = np.zeros(n)
            prev_mean = mean.copy()
            local_best_f = best_f
            local_best_x = best_x.copy()
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            no_improve = 0
            stagnation_threshold = 30

            while eval_count < max_evals and time.time() < max_time:
                if time.time() > deadline - 3:
                    break
                population = []
                fitness = []
                for i in range(lambda_):
                    if eval_count >= max_evals or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    try:
                        B = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        B = np.eye(n)
                    x_i = mean + sigma * (B @ z)
                    x_i = np.clip(x_i, -5.0, 5.0)
                    f_i = wrapped_func(x_i)
                    if f_i < local_best_f:
                        local_best_f = f_i
                        local_best_x = x_i.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                    population.append(x_i)
                    fitness.append(f_i)

                if len(population) < mu:
                    break

                # Check for stagnation
                if no_improve >= stagnation_threshold:
                    break

                idx = np.argsort(fitness)
                population = [population[i] for i in idx]
                fitness = [fitness[i] for i in idx]

                new_mean = sum(weights[i] * population[i] for i in range(mu))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (new_mean - prev_mean) / sigma
                diff = (new_mean - prev_mean) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(diff, diff) + (1 - (1 - cs) ** 2) * C)
                for i in range(mu):
                    xi = (population[i] - prev_mean) / sigma
                    C += cmu * weights[i] * np.outer(xi, xi)
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(cs / np.sqrt(n) * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                prev_mean = mean
                mean = new_mean
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-6 * np.eye(n)

            return local_best_x, local_best_f

        # First CMA-ES run from best initial point
        cmaes_budget = min(150, budget - eval_count - 50)
        if cmaes_budget > 20 and time.time() < deadline:
            cmaes_x, cmaes_f = cmaes_run(best_x.copy(), 1.5, eval_count + cmaes_budget, deadline)
            if cmaes_f < best_f:
                best_f = cmaes_f
                best_x = cmaes_x

        # --- Phase 3: GP-guided search with LCB acquisition ---
        def build_gp(X, Y):
            """Build a simple GP surrogate."""
            n_samples = len(X)
            if n_samples < 2:
                return None
            # Normalize inputs
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-8
            X_norm = (X - X_mean) / X_std
            # Normalize outputs
            Y_mean = Y.mean()
            Y_std = Y.std() + 1e-8
            Y_norm = (Y - Y_mean) / Y_std

            # Matern kernel (nu=2.5)
            def kernel(A, B, length_scale=1.0):
                dist = np.sqrt(np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2))
                return np.exp(-np.sqrt(3) * dist / length_scale) * (1 + np.sqrt(3) * dist / length_scale)

            K = kernel(X_norm, X_norm, length_scale=1.0) + 0.01 * np.eye(n_samples)
            L = np.linalg.cholesky(K)

            def predict(X_new):
                X_new_norm = (X_new - X_mean) / X_std
                K_star = kernel(X_new_norm, X_norm, length_scale=1.0)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                mu = K_star @ alpha
                v = np.linalg.solve(L, K_star.T)
                var = 1.0 - np.sum(v ** 2, axis=1)
                return mu * Y_std + Y_mean, np.sqrt(np.maximum(var, 1e-10)) * Y_std

            return predict, X_mean, X_std

        def gp_lcb_search(n_candidates, n_evals):
            nonlocal eval_count, deadline, best_x, best_f
            if len(all_x) < 5:
                return
            X = np.array(all_x)
            Y = np.array(all_f)
            predict = build_gp(X, Y)
            if predict is None:
                return
            predict_fn, X_mean, X_std = predict

            # Generate candidates
            candidates = []
            for _ in range(n_candidates):
                # Mix of random and perturbed best
                if rng.random() < 0.5:
                    cand = rng.uniform(-5.0, 5.0, dim)
                else:
                    cand = best_x + rng.normal(0, 1.0, dim)
                cand = np.clip(cand, -5.0, 5.0)
                candidates.append(cand)
            candidates = np.array(candidates)

            # Compute LCB acquisition
            mu, sigma = predict_fn(candidates)
            xi = 0.5  # exploration-exploitation balance
            lcb = mu - xi * sigma
            order = np.argsort(lcb)

            # Evaluate top candidates
            for idx in order[:n_evals]:
                if eval_count >= budget - 10 or time.time() > deadline:
                    break
                x_cand = candidates[idx]
                f_cand = wrapped_func(x_cand)
                update_best(x_cand, f_cand)

        # Run GP-guided search
        gp_rounds = 5
        for r in range(gp_rounds):
            if eval_count >= budget - 30 or time.time() > deadline:
                break
            n_cand = min(50, budget - eval_count - 30)
            n_eval = min(10, n_cand)
            if n_eval < 2:
                break
            gp_lcb_search(n_cand, n_eval)

        # --- Phase 4: Final local refinement ---
        if eval_count < budget - 10 and time.time() < deadline:
            # Run CMA-ES from current best
            cmaes_budget = min(50, budget - eval_count - 5)
            if cmaes_budget > 10:
                cmaes_x, cmaes_f = cmaes_run(best_x.copy(), 0.3, eval_count + cmaes_budget, deadline)
                if cmaes_f < best_f:
                    best_f = cmaes_f
                    best_x = cmaes_x

        # Fill remaining budget with local perturbations
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.01, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
