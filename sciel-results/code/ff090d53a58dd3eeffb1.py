"""Surrogate-assisted Bayesian Optimization with local ES refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start surrogate-assisted optimizer for 5-D optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse initialization using Halton sequence + random points
        n_init = 25
        init_points = []

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        for i in range(1, n_init + 1):
            if time.time() > deadline or calls_made >= budget:
                break
            point = []
            for d in range(dim):
                point.append(halton(i, bases[d]))
            point = np.array(point) * 10.0 - 5.0
            point += rng.uniform(-0.05, 0.05, dim)
            point = np.clip(point, -5, 5)
            init_points.append(point)

        # Add some random points
        for _ in range(5):
            if time.time() > deadline or calls_made >= budget:
                break
            point = rng.uniform(-5, 5, dim)
            init_points.append(point)

        # Evaluate all initial points
        init_results = []
        for point in init_points:
            if time.time() > deadline or calls_made >= budget:
                break
            f = evaluate(point)
            init_results.append((point, f))

        # Sort by fitness
        init_results.sort(key=lambda x: x[1])

        # Phase 2: Surrogate-assisted search with local ES refinement
        # Fit a simple Gaussian Process surrogate and use it to guide exploration
        X_obs = np.array([r[0] for r in init_results])
        Y_obs = np.array([r[1] for r in init_results])

        # Normalize Y for GP stability
        y_mean = Y_obs.mean()
        y_std = Y_obs.std() + 1e-8
        Y_norm = (Y_obs - y_mean) / y_std

        def gp_predict(X_new):
            """Simple GP prediction with RBF kernel."""
            if len(X_obs) == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new))

            # RBF kernel
            def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
                X1 = X1.reshape(-1, 1, dim)
                X2 = X2.reshape(1, -1, dim)
                sq_dist = np.sum((X1 - X2)**2, axis=2)
                return variance * np.exp(-0.5 * sq_dist / (length_scale**2))

            K = rbf_kernel(X_obs, X_obs) + 1e-6 * np.eye(len(X_obs))
            K_new = rbf_kernel(X_obs, X_new)

            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)

            mu = K_new @ K_inv @ Y_norm
            v = K_new @ K_inv @ K_new.T
            var = np.maximum(1.0 - np.diag(v), 1e-8)
            std = np.sqrt(var)

            # Denormalize
            mu = mu * y_std + y_mean
            std = std * y_std
            return mu, std

        # Phase 2a: Alternating between GP-guided exploration and local ES
        # We'll do several rounds of: pick a point from GP acquisition, then do local ES

        n_rounds = 8
        budget_per_round = (budget - calls_made) // n_rounds

        for round_idx in range(n_rounds):
            if time.time() > deadline or calls_made >= budget:
                break

            # Refit GP with all observations so far
            # (We don't track all observations explicitly, so we use init_results + best)
            # For simplicity, we'll use the top K points as our "observed" set
            K_top = min(15, len(init_results))
            top_points = init_results[:K_top]
            X_obs = np.array([r[0] for r in top_points])
            Y_obs = np.array([r[1] for r in top_points])
            y_mean = Y_obs.mean()
            y_std = Y_obs.std() + 1e-8
            Y_norm = (Y_obs - y_mean) / y_std

            # Generate candidate points: mix of GP-guided and random
            n_candidates = 10
            candidates = []

            # GP-guided candidates using Expected Improvement
            X_grid = rng.uniform(-5, 5, (50, dim))
            mu, std = gp_predict(X_grid)
            # Expected Improvement (maximization of -f, so minimization)
            # EI = (f_best - mu) * Phi((f_best - mu)/std) + std * phi((f_best - mu)/std)
            from scipy.stats import norm
            z = (best_f - mu) / (std + 1e-8)
            ei = (best_f - mu) * norm.cdf(z) + std * norm.pdf(z)
            # Pick top candidates by EI
            ei_indices = np.argsort(-ei)[:n_candidates]
            for idx in ei_indices:
                candidates.append(X_grid[idx])

            # Add some random candidates
            for _ in range(5):
                candidates.append(rng.uniform(-5, 5, dim))

            # Evaluate candidates and pick the best
            best_candidate = None
            best_candidate_f = np.inf
            for cand in candidates:
                if time.time() > deadline or calls_made >= budget:
                    break
                cand = np.clip(cand, -5, 5)
                f = evaluate(cand)
                if f < best_candidate_f:
                    best_candidate_f = f
                    best_candidate = cand

            if best_candidate is None:
                break

            # Phase 2b: Local ES refinement from the best candidate
            x = best_candidate.copy()
            fx = best_candidate_f
            sigma = 0.5
            es_calls = 0
            max_es_calls = budget_per_round // 2
            stagnation = 0
            max_stagnation = 15

            while es_calls < max_es_calls:
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                es_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    stagnation = 0
                else:
                    sigma *= 0.92
                    stagnation += 1

                sigma = np.clip(sigma, 0.01, 5.0)

                if stagnation >= max_stagnation:
                    break

            # Update init_results with the best found in this round
            init_results.append((x, fx))
            init_results.sort(key=lambda r: r[1])

        # Phase 3: Final polish on global best
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.1
            polish_calls = 0
            max_polish = min(20, budget - calls_made)

            for _ in range(max_polish):
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x_polish + sigma_p * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                polish_calls += 1

                if fy < f_polish:
                    x_polish, f_polish = y, fy
                    sigma_p *= 1.1
                else:
                    sigma_p *= 0.9

                sigma_p = np.clip(sigma_p, 0.005, 1.0)

            if f_polish < best_f:
                best_x = x_polish.copy()
                best_f = f_polish

        return best_x, best_f
# EVOLVE-BLOCK-END
