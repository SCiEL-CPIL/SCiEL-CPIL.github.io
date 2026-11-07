"""Gaussian Process model-based optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-based Bayesian optimization with Gaussian Process regression."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)

        # Phase 1: Initial structured sampling to build GP training set
        points = []
        values = []

        # Origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        points.append(x0)
        values.append(f0)
        calls += 1

        # Probe along each axis (both directions)
        probe_len = 3.0
        for i in range(dim):
            for sign in [1, -1]:
                if check_deadline() or calls >= budget:
                    break
                x = np.zeros(dim)
                x[i] = sign * probe_len
                fx = func(x)
                points.append(x)
                values.append(fx)
                calls += 1

        # Additional random points for better GP coverage
        n_random = min(15, budget - calls - 10)
        for _ in range(n_random):
            if check_deadline() or calls >= budget:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)
            calls += 1

        points_arr = np.array(points)
        values_arr = np.array(values)

        best_x = points_arr[np.argmin(values_arr)].copy()
        best_f = float(np.min(values_arr))

        # Phase 2: Gaussian Process optimization loop
        # GP hyperparameters
        length_scales = np.ones(dim) * 1.5
        signal_variance = 1.0
        noise_variance = 1e-6

        def gp_predict(X_pred):
            """Compute GP mean and std at X_pred given training data."""
            n_train = len(values_arr)
            n_pred = X_pred.shape[0]
            
            # Compute K(X_train, X_train)
            diff_train = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
            sq_dist_train = np.sum(diff_train ** 2 / length_scales[np.newaxis, np.newaxis, :], axis=2)
            K_train = signal_variance * np.exp(-0.5 * sq_dist_train) + noise_variance * np.eye(n_train)
            
            # Compute K(X_pred, X_train)
            diff_pred = X_pred[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
            sq_dist_pred = np.sum(diff_pred ** 2 / length_scales[np.newaxis, np.newaxis, :], axis=2)
            K_pred_train = signal_variance * np.exp(-0.5 * sq_dist_pred)
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, values_arr))
            except:
                # Add jitter if Cholesky fails
                K_train += 1e-6 * np.eye(n_train)
                try:
                    L = np.linalg.cholesky(K_train)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, values_arr))
                except:
                    # Fall back to pseudo-inverse
                    alpha = np.linalg.pinv(K_train) @ values_arr
                    L = None
            
            # Predictions
            mean_pred = K_pred_train @ alpha
            
            if L is not None:
                v = np.linalg.solve(L, K_pred_train.T)
                var_pred = signal_variance - np.sum(v ** 2, axis=0)
            else:
                # Approximate variance using inverse
                K_inv = np.linalg.pinv(K_train)
                v = K_inv @ K_pred_train.T
                var_pred = signal_variance - np.sum(K_pred_train * v, axis=1)
            
            std_pred = np.sqrt(np.maximum(var_pred, 1e-12))
            
            return mean_pred, std_pred

        # Expected Improvement acquisition function
        def expected_improvement(X_pred, current_best):
            """Compute EI for minimization."""
            mean, std = gp_predict(X_pred)
            improvement = current_best - mean
            eps = 1e-8
            std_safe = np.maximum(std, eps)
            z = improvement / std_safe
            # Normal CDF and PDF
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + std_safe * norm.pdf(z)
            # Handle case where improvement <= 0
            ei = np.where(improvement > 0, ei, 0.0)
            return ei

        # GP optimization loop
        gp_candidates_per_iter = 20
        max_gp_iters = min(40, (budget - calls) // gp_candidates_per_iter)

        for gp_iter in range(max_gp_iters):
            if check_deadline() or calls >= budget:
                break

            # Generate candidate points
            candidates = []
            
            # Local candidates around current best
            n_local = 8
            for _ in range(n_local):
                x_cand = best_x + rng.standard_normal(dim) * 0.5
                x_cand = np.clip(x_cand, -5.0, 5.0)
                candidates.append(x_cand)
            
            # Random candidates
            n_rand = gp_candidates_per_iter - n_local
            for _ in range(n_rand):
                x_cand = rng.uniform(-5.0, 5.0, dim)
                candidates.append(x_cand)
            
            candidates = np.array(candidates)
            
            # Compute EI for all candidates
            try:
                ei_values = expected_improvement(candidates, best_f)
            except:
                # Fallback: just pick a random point near best
                x_next = best_x + rng.standard_normal(dim) * 0.3
                x_next = np.clip(x_next, -5.0, 5.0)
                f_next = func(x_next)
                calls += 1
                if f_next < best_f:
                    best_x = x_next.copy()
                    best_f = f_next
                points_arr = np.vstack([points_arr, x_next])
                values_arr = np.append(values_arr, f_next)
                continue

            # Select best candidate by EI
            idx = np.argmax(ei_values)
            x_next = candidates[idx]
            f_next = func(x_next)
            calls += 1
            
            # Update training data
            points_arr = np.vstack([points_arr, x_next])
            values_arr = np.append(values_arr, f_next)
            
            # Update best
            if f_next < best_f:
                best_x = x_next.copy()
                best_f = f_next

        # Phase 3: Local polish with (1+1)-ES
        if not check_deadline() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.3
            successes = 0
            total = 0
            eps = 1e-10

            remaining = budget - calls
            for _ in range(remaining):
                if check_deadline() or calls >= budget:
                    break
                y = x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1

                if fy < fx - eps:
                    x = y
                    fx = fy
                    successes += 1
                total += 1

                if total >= 5:
                    sr = successes / total
                    if sr > 0.3:
                        sigma_local *= 1.3
                    elif sr < 0.2:
                        sigma_local *= 0.7
                    successes = 0
                    total = 0

                if sigma_local < 0.01:
                    sigma_local = 0.3

            if fx < best_f:
                best_x = x
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
