"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Bayesian Optimization with GP surrogate for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _latin_hypercube(self, rng, n_samples, dim):
        """Generate n_samples points via Latin Hypercube Sampling in [-5, 5]^dim."""
        points = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = rng.permutation(n_samples)
            points[:, d] = (perm + 0.5) / n_samples * 10.0 - 5.0
        return points

    def _rbf_kernel(self, X, Y, length_scale, variance):
        """Compute RBF kernel matrix between X and Y."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
        sq_dists = np.maximum(sq_dists, 0)
        return variance * np.exp(-0.5 * sq_dists / (length_scale ** 2))

    def _fit_gp(self, X, y, rng):
        """Fit GP hyperparameters via simple grid search on log scale."""
        n, d = X.shape
        
        # Normalize inputs to [0, 1] for better conditioning
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        x_range = x_max - x_min
        x_range[x_range < 1e-8] = 1.0
        X_norm = (X - x_min) / x_range
        
        # Normalize outputs
        y_mean = y.mean()
        y_std = y.std()
        if y_std < 1e-8:
            y_std = 1.0
        y_norm = (y - y_mean) / y_std
        
        best_score = -np.inf
        best_ls = 0.5
        best_var = 1.0
        best_noise = 1e-4
        
        # Grid search over hyperparameters
        for ls in [0.1, 0.2, 0.5, 1.0, 2.0]:
            for var in [0.5, 1.0, 2.0]:
                for noise in [1e-4, 1e-3, 1e-2]:
                    try:
                        K = self._rbf_kernel(X_norm, X_norm, ls, var) + noise * np.eye(n)
                        L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                        
                        # Marginal log likelihood
                        mll = -0.5 * y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                        
                        if mll > best_score:
                            best_score = mll
                            best_ls = ls
                            best_var = var
                            best_noise = noise
                    except np.linalg.LinAlgError:
                        continue
        
        return best_ls, best_var, best_noise, x_min, x_range, y_mean, y_std

    def _gp_predict(self, X_test, X_train, y_train, ls, var, noise, x_min, x_range, y_mean, y_std):
        """Predict mean and variance at X_test using fitted GP."""
        X_test_norm = (X_test - x_min) / x_range
        X_train_norm = (X_train - x_min) / x_range
        y_train_norm = (y_train - y_mean) / y_std
        
        K_train = self._rbf_kernel(X_train_norm, X_train_norm, ls, var) + noise * np.eye(len(X_train))
        K_test = self._rbf_kernel(X_test_norm, X_train_norm, ls, var)
        
        try:
            L = np.linalg.cholesky(K_train + 1e-8 * np.eye(len(X_train)))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train_norm))
            
            mean_norm = K_test @ alpha
            v = np.linalg.solve(L, K_test.T)
            var_norm = var - np.sum(v**2, axis=0)
            var_norm = np.maximum(var_norm, 1e-8)
            
            mean = mean_norm * y_std + y_mean
            var = var_norm * y_std**2
        except np.linalg.LinAlgError:
            mean = np.full(X_test.shape[0], y_train.mean())
            var = np.full(X_test.shape[0], y_train.var() + 1e-4)
        
        return mean, var

    def _expected_improvement(self, mean, var, f_best, xi=0.01):
        """Compute Expected Improvement acquisition function."""
        var = np.maximum(var, 1e-8)
        sigma = np.sqrt(var)
        z = (f_best - mean - xi) / sigma
        from scipy.stats import norm
        ei = (f_best - mean - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 270.0

        calls = 0

        def safe_func(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError()
            if calls >= budget:
                raise TimeoutError()
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            return func(x)

        # Phase 1: LHS initialization - 15 points for broad exploration
        n_lhs = min(15, budget // 20)
        lhs_points = self._latin_hypercube(rng, n_lhs, dim)
        lhs_values = np.array([safe_func(p) for p in lhs_points])
        
        X_obs = lhs_points.copy()
        y_obs = lhs_values.copy()
        
        best_idx = np.argmin(y_obs)
        x_best = X_obs[best_idx].copy()
        f_best = y_obs[best_idx]

        if time.time() > deadline or calls >= budget:
            return x_best, f_best

        # Phase 2: Bayesian Optimization with GP surrogate
        # Use EI acquisition with random candidate sampling
        
        bo_budget = budget - calls
        n_candidates = 200  # Number of candidate points to evaluate EI on
        
        for iteration in range(bo_budget):
            if time.time() > deadline or calls >= budget:
                break
            
            # Fit GP to current observations
            if len(X_obs) >= 5:
                ls, var, noise, x_min, x_range, y_mean, y_std = self._fit_gp(X_obs, y_obs, rng)
            else:
                # If too few points, use simple distance-based heuristic
                ls, var, noise = 0.5, 1.0, 1e-3
                x_min = X_obs.min(axis=0)
                x_max = X_obs.max(axis=0)
                x_range = x_max - x_min
                x_range[x_range < 1e-8] = 1.0
                y_mean = y_obs.mean()
                y_std = max(y_obs.std(), 1e-8)
            
            # Generate candidate points: mix of random and local perturbations
            candidates = np.zeros((n_candidates, dim))
            
            # 60% random uniform
            n_rand = int(n_candidates * 0.6)
            candidates[:n_rand] = rng.uniform(-5, 5, size=(n_rand, dim))
            
            # 40% local perturbations around best points
            n_local = n_candidates - n_rand
            for i in range(n_local):
                # Pick a random observed point (weighted towards better ones)
                weights = np.exp(-0.5 * (y_obs - f_best) / max(f_best * 0.1, 1e-8))
                weights = weights / weights.sum()
                idx = rng.choice(len(X_obs), p=weights)
                scale = 0.5 * np.exp(-0.05 * iteration)  # Decreasing scale
                candidates[n_rand + i] = np.clip(X_obs[idx] + scale * rng.standard_normal(dim), -5, 5)
            
            # Evaluate EI at candidates
            mean, var = self._gp_predict(candidates, X_obs, y_obs, ls, var, noise, x_min, x_range, y_mean, y_std)
            ei = self._expected_improvement(mean, var, f_best, xi=0.01)
            
            # Select best candidate
            best_candidate_idx = np.argmax(ei)
            x_new = candidates[best_candidate_idx]
            
            # Evaluate
            f_new = safe_func(x_new)
            
            # Update observations
            X_obs = np.vstack([X_obs, x_new.reshape(1, -1)])
            y_obs = np.append(y_obs, f_new)
            
            # Update best
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()

        # Phase 3: Nelder-Mead polish with remaining budget
        remaining = budget - calls
        if remaining > 10:
            nm_maxiter = max(1, remaining // (dim + 1))
            nm_maxiter = min(nm_maxiter, 100)

            try:
                result = minimize(
                    safe_func,
                    x_best,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_maxiter,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < f_best:
                    x_best, f_best = result.x, result.fun
            except Exception:
                pass

        return x_best, f_best
# EVOLVE-BLOCK-END
