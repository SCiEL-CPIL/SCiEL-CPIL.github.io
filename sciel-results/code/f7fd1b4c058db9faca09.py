"""Bayesian Optimization surrogate (Gaussian Process) with Expected Improvement for BBOB f8."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.distance import cdist


class Optimizer:
    """Gaussian Process Bayesian Optimizer with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_new, X, y, length_scale, noise):
        """
        Compute GP mean and variance at X_new given training data X, y.
        Uses a simple RBF kernel.
        """
        n = X.shape[0]
        m = X_new.shape[0]
        
        # Compute kernel matrix K(X, X)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        K = np.exp(-0.5 * sq_dist / (length_scale ** 2)) + noise * np.eye(n)
        
        # Precompute Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter if Cholesky fails
            K += 1e-6 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback to eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(K)
                eigvals = np.maximum(eigvals, 1e-8)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # Compute alpha = K^{-1} y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        # Compute K(X_new, X)
        diff_new = X_new[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist_new = np.sum(diff_new ** 2, axis=-1)
        K_star = np.exp(-0.5 * sq_dist_new / (length_scale ** 2))
        
        # Predict mean
        mean = K_star @ alpha
        
        # Predict variance
        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        
        return mean, var

    def _expected_improvement(self, x, y_best, mean, var, xi=0.01):
        """Compute Expected Improvement acquisition function."""
        sigma = np.sqrt(var)
        z = (y_best - mean - xi) / sigma
        ei = (y_best - mean - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def _sobol_sample(self, n, dim, rng):
        """Generate quasi-random Sobol-like samples using Halton sequence approximation."""
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                # Halton sequence in base (d+2)
                base = d + 2
                x = 0.0
                f = 1.0 / base
                i_val = i + 1
                while i_val > 0:
                    x += f * (i_val % base)
                    i_val //= base
                    f /= base
                points[i, d] = x
        # Scale to [-5, 5]
        points = points * 10 - 5
        return points

    def _local_optimize(self, x0, func, dim, max_iter=50):
        """Run a local optimization from x0 using Nelder-Mead."""
        try:
            result = minimize(
                func, x0, method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            return result.x, result.fun
        except Exception:
            return x0, func(x0)

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Initial exploration with quasi-random sampling
        n_init = min(15, budget)
        X = []
        y = []
        
        # Add center
        x_center = np.zeros(dim)
        X.append(x_center)
        y.append(func(x_center))
        
        # Add quasi-random points
        sobol_points = self._sobol_sample(n_init - 1, dim, rng)
        for i in range(n_init - 1):
            x = sobol_points[i]
            X.append(x)
            y.append(func(x))
        
        X = np.array(X)
        y = np.array(y)
        
        # Adaptive length scale and noise
        length_scale = 2.0
        noise = 1e-5
        
        # Track best
        y_best = np.min(y)
        x_best = X[np.argmin(y)].copy()
        
        # Call count tracking
        calls_made = n_init
        
        # Bayesian optimization loop
        while calls_made < budget:
            # Phase determination: early (exploration), mid (balance), late (exploitation)
            progress = (calls_made - n_init) / max(1, budget - n_init)
            
            # Adaptive hyperparameters based on progress
            if progress < 0.3:
                # Early: broad exploration
                length_scale = max(1.5, 2.0 * (1 - progress * 0.5))
                n_candidates = 300
                xi = 0.05
            elif progress < 0.6:
                # Mid: balance
                length_scale = max(1.0, 1.5 - progress * 0.5)
                n_candidates = 200
                xi = 0.02
            else:
                # Late: exploitation
                length_scale = max(0.5, 1.0 - (progress - 0.6) * 1.0)
                n_candidates = 150
                xi = 0.01
            
            # Generate candidate points
            candidates = []
            
            # Random candidates
            n_random = int(n_candidates * 0.4)
            candidates.append(rng.uniform(-5, 5, (n_random, dim)))
            
            # Local perturbations around best point
            if calls_made > n_init + 10:
                n_local = int(n_candidates * 0.4)
                local_sigma = max(0.3, length_scale * 0.7)
                local_candidates = x_best + local_sigma * rng.standard_normal((n_local, dim))
                local_candidates = np.clip(local_candidates, -5, 5)
                candidates.append(local_candidates)
            
            # Points near other good solutions
            if len(y) > 5 and calls_made > n_init + 20:
                n_near_good = int(n_candidates * 0.2)
                # Get top 3 best points
                top_k = min(3, len(y))
                top_indices = np.argsort(y)[:top_k]
                near_good = []
                for idx in top_indices:
                    sigma = max(0.5, length_scale * 0.5)
                    near_good.append(X[idx] + sigma * rng.standard_normal(dim))
                near_good = np.array(near_good)
                candidates.append(near_good)
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, -5, 5)
            
            # Evaluate acquisition function
            mean, var = self._gp_predict(candidates, X, y, length_scale, noise)
            ei = self._expected_improvement(candidates, y_best, mean, var, xi)
            
            # Select best candidate
            best_idx = np.argmax(ei)
            x_new = candidates[best_idx]
            
            # Evaluate function
            f_new = func(x_new)
            calls_made += 1
            
            # Update data
            X = np.vstack([X, x_new])
            y = np.append(y, f_new)
            
            # Update best
            if f_new < y_best:
                y_best = f_new
                x_best = x_new.copy()
                # When we find improvement, run local optimization if we have budget
                if calls_made < budget - 10 and progress > 0.3:
                    x_local, f_local = self._local_optimize(x_best, func, dim, max_iter=20)
                    # Count the local optimization calls (approximate)
                    local_calls = min(20, budget - calls_made)
                    calls_made += local_calls
                    if f_local < y_best:
                        y_best = f_local
                        x_best = x_local.copy()
                        # Add the local optimum to our data
                        X = np.vstack([X, x_best])
                        y = np.append(y, y_best)
        
        # Final local optimization if we have budget remaining
        if calls_made < budget - 5:
            x_final, f_final = self._local_optimize(x_best, func, dim, max_iter=30)
            if f_final < y_best:
                y_best = f_final
                x_best = x_final.copy()
        
        return x_best, y_best
# EVOLVE-BLOCK-END
