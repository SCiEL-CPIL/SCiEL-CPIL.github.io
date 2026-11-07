import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            calls += 1
            try:
                f = func(x)
                if not np.isfinite(f):
                    f = float('inf')
            except Exception:
                f = float('inf')

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead implementation - robust and efficient
        def run_nelder_mead(x_start, max_calls):
            nonlocal calls, best_f, best_x
            local_best_x = x_start.copy()
            local_best_f = evaluate(x_start)
            local_calls = 1

            if local_calls >= max_calls or time.time() > deadline:
                return local_best_x, local_best_f, local_calls

            # Initial simplex
            simplex = [x_start.copy()]
            for i in range(dim):
                x_i = x_start.copy()
                # Use a moderate step size
                step = 0.5
                x_i[i] += step
                simplex.append(x_i)

            simplex_f = [local_best_f]
            for i in range(1, dim + 1):
                simplex_f.append(evaluate(simplex[i]))
                local_calls += 1
                if simplex_f[i] < local_best_f:
                    local_best_f = simplex_f[i]
                    local_best_x = simplex[i].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            min_step = 1e-10

            while local_calls < max_calls and time.time() < deadline:
                # Sort simplex
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]

                if simplex_f[0] < local_best_f:
                    local_best_f = simplex_f[0]
                    local_best_x = simplex[0].copy()

                # Check convergence
                if local_calls >= max_calls - 1:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)
                local_calls += 1

                if f_r < local_best_f:
                    local_best_f = f_r
                    local_best_x = x_r.copy()

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    local_calls += 1
                    if f_e < local_best_f:
                        local_best_f = f_e
                        local_best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    local_calls += 1
                    if f_c < local_best_f:
                        local_best_f = f_c
                        local_best_x = x_c.copy()
                    if f_c < f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = evaluate(simplex[i])
                            local_calls += 1
                            if simplex_f[i] < local_best_f:
                                local_best_f = simplex_f[i]
                                local_best_x = simplex[i].copy()

                        # Check if shrinkage was effective
                        if np.max(np.abs(simplex[1] - simplex[0])) < min_step:
                            break

            return local_best_x, local_best_f, local_calls

        # Phase 1: Light exploration with diverse starts
        # Use a small number of diverse initial points
        rng = np.random.default_rng(42)
        initial_starts = [
            np.zeros(dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-4, 4, dim),
        ]

        X_samples = []
        f_samples = []

        for x0 in initial_starts:
            if calls >= budget - 150:  # Reserve budget for Nelder-Mead
                break
            if time.time() > deadline:
                break

            x0 = np.clip(x0, -5, 5)
            f0 = evaluate(x0)
            X_samples.append(x0)
            f_samples.append(f0)

        # Phase 2: Use a lightweight GP to guide a few more exploratory evaluations
        # Keep this phase short to preserve budget for local optimization
        gp_iterations = 0
        max_gp_iterations = 5  # Very few GP-guided evaluations

        while (calls < budget - 150 and
               time.time() < deadline and
               gp_iterations < max_gp_iterations and
               len(X_samples) >= 3):

            X_arr = np.array(X_samples)
            f_arr = np.array(f_samples)

            # Fit simple GP
            L = 1.0  # length scale
            sigma_f = np.std(f_arr) + 1e-6

            def rbf_kernel(X1, X2, L=L, sigma_f=sigma_f):
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                K = sigma_f**2 * np.exp(-0.5 * sq_dists / L**2)
                return K

            K = rbf_kernel(X_arr, X_arr) + 1e-6 * np.eye(len(X_arr))

            # Generate candidate points
            n_candidates = 50
            candidates = rng.uniform(-4.5, 4.5, (n_candidates, dim))

            K_x = rbf_kernel(X_arr, candidates)
            K_xx = rbf_kernel(candidates, candidates)

            try:
                K_inv = np.linalg.inv(K)
                f_mean = K_x.T @ K_inv @ f_arr
                f_var = np.diag(K_xx) - np.sum(K_x.T @ K_inv @ K_x, axis=0)
                f_var = np.maximum(f_var, 1e-10)
                f_std = np.sqrt(f_var)
            except np.linalg.LinAlgError:
                # Fallback to random point
                x_new = rng.uniform(-4, 4, dim)
                f_new = evaluate(x_new)
                X_samples.append(x_new)
                f_samples.append(f_new)
                gp_iterations += 1
                continue

            # Expected Improvement
            from scipy.stats import norm
            current_best = np.min(f_arr)
            Z = (current_best - f_mean) / (f_std + 1e-10)
            EI = (current_best - f_mean) * norm.cdf(Z) + f_std * norm.pdf(Z)

            # Select point with max EI
            idx_best = np.argmax(EI)
            x_new = candidates[idx_best]
            f_new = evaluate(x_new)

            X_samples.append(x_new)
            f_samples.append(f_new)

            gp_iterations += 1

        # Phase 3: Exploit with Nelder-Mead from the best point
        if calls < budget - 10 and time.time() < deadline:
            remaining_budget = budget - calls
            x_final, f_final, _ = run_nelder_mead(best_x, remaining_budget)

        return best_x, best_f
