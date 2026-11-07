# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization-based multi-objective optimizer using GP surrogate."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])

        # Data storage
        X_data = []  # List of decision vectors
        F_data = []  # List of objective vectors

        used = 0

        # Initial random sampling
        n_init = 20
        X_init = lo + span * np.random.random((n_init, dim))
        for i in range(n_init):
            f = problem(X_init[i])
            X_data.append(X_init[i])
            F_data.append(f)
            used += 1
            if time.time() > deadline:
                break

        if used >= budget:
            return np.array(X_data)

        # GP Surrogate Model
        class GPMultiObjective:
            def __init__(self, dim, n_obj):
                self.dim = dim
                self.n_obj = n_obj
                self.models = [None] * n_obj
                self.X = None
                self.Y = None

            def fit(self, X, Y):
                self.X = X
                self.Y = Y
                n = len(X)
                for j in range(self.n_obj):
                    self.models[j] = self._fit_gp(X, Y[:, j])

            def _fit_gp(self, X, y):
                # Normalize y
                y_mean = np.mean(y)
                y_std = np.std(y)
                if y_std < 1e-10:
                    y_std = 1.0
                y_norm = (y - y_mean) / y_std

                n = len(X)
                # Use RBF kernel
                K = self._rbf_kernel(X, X)
                # Add noise for numerical stability
                K += 1e-6 * np.eye(n)

                # Solve for weights
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                except np.linalg.LinAlgError:
                    # Fallback to pinv
                    alpha = np.linalg.pinv(K) @ y_norm

                return {'X': X, 'y_norm': y_norm, 'y_mean': y_mean, 'y_std': y_std,
                        'alpha': alpha, 'K': K}

            def _rbf_kernel(self, X1, X2, length_scale=0.5):
                # Compute pairwise squared distances
                sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-0.5 * sq_dists / (length_scale ** 2))

            def predict(self, X_new):
                n_new = len(X_new)
                means = np.zeros((n_new, self.n_obj))
                stds = np.zeros((n_new, self.n_obj))

                for j in range(self.n_obj):
                    model = self.models[j]
                    if model is None:
                        means[:, j] = 0
                        stds[:, j] = 1
                        continue

                    K_s = self._rbf_kernel(X_new, model['X'])
                    mu_norm = K_s @ model['alpha']
                    means[:, j] = mu_norm * model['y_std'] + model['y_mean']

                    # Variance calculation (approximate)
                    # v(x) = 1 - k(x, X) K^{-1} k(X, x)
                    # We use the precomputed K^{-1} implicitly via alpha, but for variance we need K^{-1}
                    # Let's compute K^{-1} once during fit for efficiency? Or just approximate.
                    # For speed, let's use a simple approximation or recompute.
                    # Actually, computing K^{-1} is O(n^3). With n=200, it's 8e6 ops, which is fine.
                    # But we do it for every prediction.
                    # Let's store K_inv in the model.
                    # We didn't store K_inv. Let's modify _fit_gp to store it.
                    # Since we can't easily modify the class definition inside __call__ without breaking structure,
                    # let's just compute it here if not stored, or use a simpler variance estimate.
                    # Given the tight budget, let's use a simpler acquisition function that doesn't require full GP variance,
                    # or we compute K_inv in fit.

                    # Let's refactor _fit_gp to store K_inv.
                    # Wait, I can't easily change the class definition mid-stream in the SEARCH/REPLACE if it's complex.
                    # Let's just compute K_inv in _fit_gp and store it.

                    # For now, let's assume we have K_inv. I will update _fit_gp in the code below.
                    pass

                return means, stds

        # Refactor GP to store K_inv
        class GPMultiObjective:
            def __init__(self, dim, n_obj):
                self.dim = dim
                self.n_obj = n_obj
                self.models = [None] * n_obj
                self.X = None
                self.Y = None

            def fit(self, X, Y):
                self.X = X
                self.Y = Y
                n = len(X)
                for j in range(self.n_obj):
                    self.models[j] = self._fit_gp(X, Y[:, j])

            def _fit_gp(self, X, y):
                y_mean = np.mean(y)
                y_std = np.std(y)
                if y_std < 1e-10:
                    y_std = 1.0
                y_norm = (y - y_mean) / y_std

                n = len(X)
                K = self._rbf_kernel(X, X)
                K += 1e-6 * np.eye(n)

                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                    K_inv = np.linalg.inv(K)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.pinv(K) @ y_norm
                    K_inv = np.linalg.pinv(K)

                return {'X': X, 'y_norm': y_norm, 'y_mean': y_mean, 'y_std': y_std,
                        'alpha': alpha, 'K': K, 'K_inv': K_inv}

            def _rbf_kernel(self, X1, X2, length_scale=0.5):
                sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-0.5 * sq_dists / (length_scale ** 2))

            def predict(self, X_new):
                n_new = len(X_new)
                means = np.zeros((n_new, self.n_obj))
                stds = np.zeros((n_new, self.n_obj))

                for j in range(self.n_obj):
                    model = self.models[j]
                    if model is None:
                        means[:, j] = 0
                        stds[:, j] = 1
                        continue

                    K_s = self._rbf_kernel(X_new, model['X'])
                    mu_norm = K_s @ model['alpha']
                    means[:, j] = mu_norm * model['y_std'] + model['y_mean']

                    # Variance: 1 - K_s @ K_inv @ K_s.T
                    v_norm = 1.0 - np.sum((K_s @ model['K_inv']) * K_s, axis=1)
                    v_norm = np.maximum(v_norm, 0)
                    stds[:, j] = np.sqrt(v_norm) * model['y_std']

                return means, stds

        gp = GPMultiObjective(dim, n_obj)

        # Initial fit
        if used > 0:
            X_arr = np.array(X_data)
            F_arr = np.array(F_data)
            gp.fit(X_arr, F_arr)

        # Acquisition Function: Expected Hypervolume Improvement (EHVI) approximation
        # Since exact EHVI is complex, we use a simplified version or a scalarization-based EI.
        # For simplicity and speed, let's use a combination of:
        # 1. Distance to reference point (exploitation)
        # 2. Uncertainty (exploration)
        # Or simply: Minimize the predicted value of the sum of objectives, plus a bonus for uncertainty.
        # Let's try: Score = -sum(predicted_f) + kappa * sqrt(sum(predicted_var))
        # This is a simple EI-like function.

        kappa = 0.5

        while used < budget:
            if time.time() > deadline:
                break

            # Generate candidate points
            n_candidates = 50
            X_cand = lo + span * np.random.random((n_candidates, dim))

            # Predict mean and std
            means, stds = gp.predict(X_cand)

            # Calculate acquisition score
            # We want to minimize objectives, so lower mean is better.
            # Higher std is better for exploration.
            # Score = -sum(means) + kappa * sum(stds)
            scores = -np.sum(means, axis=1) + kappa * np.sum(stds, axis=1)

            # Select best candidate
            best_idx = np.argmax(scores)
            x_new = X_cand[best_idx]

            # Evaluate
            f_new = problem(x_new)
            X_data.append(x_new)
            F_data.append(f_new)
            used += 1

            # Refit GP
            X_arr = np.array(X_data)
            F_arr = np.array(F_data)
            gp.fit(X_arr, F_arr)

        return np.array(X_data)

# EVOLVE-BLOCK-END
