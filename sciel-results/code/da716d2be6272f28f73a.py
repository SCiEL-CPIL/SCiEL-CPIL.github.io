"""Bayesian Optimization with GP surrogate for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """
    Bayesian Optimization using a Gaussian Process surrogate model.
    
    Strategy:
    1. Initial Latin Hypercube sampling for diverse exploration.
    2. GP surrogate with Matern 5/2 kernel to model the landscape.
    3. Expected Improvement acquisition function to balance exploration/exploitation.
    4. Final local polish with Nelder-Mead if budget remains.
    
    This approach is particularly suited for low-dimensional (d=5) black-box
    optimization with limited budget (500) where the landscape may be complex
    (rotated, shifted, offset).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _matern52_kernel(self, X1, X2, length_scale, sigma_f):
        """Matern 5/2 kernel."""
        # Compute pairwise distances
        d = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2))
        sqrt3d = np.sqrt(3.0) * d / length_scale
        K = sigma_f ** 2 * (1 + sqrt3d + (sqrt3d ** 2) / 3.0) * np.exp(-sqrt3d)
        return K

    def _fit_gp(self, X, y, length_scale, sigma_f, noise_var):
        """Fit GP and return K matrix and alpha vector."""
        K = self._matern52_kernel(X, X, length_scale, sigma_f)
        K += noise_var * np.eye(len(X))
        # Add jitter for numerical stability
        K += 1e-8 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add more jitter if Cholesky fails
            K += 1e-6 * np.eye(len(X))
            L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return L, alpha

    def _gp_predict(self, X_new, X, y, L, alpha, length_scale, sigma_f, noise_var):
        """Predict mean and variance at X_new."""
        K_s = self._matern52_kernel(X_new, X, length_scale, sigma_f)
        K_ss = self._matern52_kernel(X_new, X_new, length_scale, sigma_f)
        
        v = np.linalg.solve(L, K_s.T)
        mu = K_s @ alpha
        var = np.diag(K_ss) - np.sum(v ** 2, axis=1)
        var = np.maximum(var, 0.0)
        return mu, np.sqrt(var)

    def _expected_improvement(self, mu, sigma, f_min, xi=0.01):
        """Compute Expected Improvement."""
        sigma = np.maximum(sigma, 1e-10)
        z = (f_min - mu - xi) / sigma
        ei = (f_min - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def _latin_hypercube_sample(self, n, dim, bounds=(-5.0, 5.0)):
        """Generate Latin Hypercube samples."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            samples[:, d] = (perm + np.random.rand(n)) / n
            samples[:, d] = bounds[0] + samples[:, d] * (bounds[1] - bounds[0])
        return samples

    def __call__(self, func):
        calls = 0
        max_calls = self.budget
        dim = self.dim
        rng = np.random.default_rng()

        def safe_call(x):
            nonlocal calls
            if calls >= max_calls:
                return np.inf
            calls += 1
            return func(x)

        # Phase 1: Initial exploration with Latin Hypercube
        n_initial = min(30, max(10, self.budget // 15))
        X_init = self._latin_hypercube_sample(n_initial, dim)
        
        X = []
        y = []
        best_x = None
        best_f = np.inf

        for x in X_init:
            f = safe_call(x)
            X.append(x)
            y.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if calls >= max_calls:
            return best_x, best_f

        # Phase 2: Bayesian Optimization loop
        # Allocate remaining budget for BO iterations
        remaining = max_calls - calls
        
        # We'll do BO iterations until budget is nearly exhausted
        # Leave ~10% for final local polish
        bo_budget = int(remaining * 0.85)
        polish_budget = remaining - bo_budget

        # GP hyperparameters (tuned for this problem)
        length_scale = 1.5
        sigma_f = 1.0
        noise_var = 1e-6

        # Track best GP parameters
        best_ls = length_scale
        best_sf = sigma_f
        best_noise = noise_var

        for it in range(bo_budget):
            if calls >= max_calls:
                break

            # Fit GP
            X_arr = np.array(X)
            y_arr = np.array(y)
            
            # Standardize y for better GP conditioning
            y_mean = np.mean(y_arr)
            y_std = np.std(y_arr) + 1e-10
            y_norm = (y_arr - y_mean) / y_std

            L, alpha = self._fit_gp(X_arr, y_norm, best_ls, best_sf, best_noise)

            # Candidate points for acquisition
            # Sample candidates from a grid + random points around current best
            n_candidates = 200
            candidates = []
            
            # Random candidates
            for _ in range(n_candidates // 2):
                candidates.append(rng.uniform(-5, 5, dim))
            
            # Candidates around current best
            for _ in range(n_candidates // 4):
                candidates.append(best_x + rng.standard_normal(dim) * 0.5)
            
            # Grid-like candidates (coarse)
            for _ in range(n_candidates // 4):
                x = rng.uniform(-5, 5, dim)
                candidates.append(x)

            candidates = np.array(candidates)

            # Compute EI for all candidates
            mu, sigma = self._gp_predict(candidates, X_arr, y_norm, L, alpha, best_ls, best_sf, best_noise)
            
            # Convert back to original scale
            mu_orig = mu * y_std + y_mean
            sigma_orig = sigma * y_std

            ei = self._expected_improvement(mu_orig, sigma_orig, best_f, xi=0.01)
            idx = np.argmax(ei)
            x_next = candidates[idx]
            x_next = np.clip(x_next, -5, 5)

            # Evaluate
            f_next = safe_call(x_next)
            X.append(x_next)
            y.append(f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()

            # Occasionally re-tune GP hyperparameters (simple approach)
            if it % 20 == 19 and it > 0:
                # Try a few different length scales
                for ls in [0.8, 1.2, 1.8, 2.5]:
                    try:
                        L_test, _ = self._fit_gp(X_arr, y_norm, ls, best_sf, best_noise)
                        # Use log-likelihood as criterion
                        ll = -0.5 * (len(y_arr) * np.log(2 * np.pi) + np.sum(np.log(np.diag(self._matern52_kernel(X_arr, X_arr, ls, best_sf)) + best_noise * np.eye(len(y_arr)))) + y_norm @ np.linalg.solve(self._matern52_kernel(X_arr, X_arr, ls, best_sf) + best_noise * np.eye(len(y_arr)), y_norm))
                        if it == 20 or ll > -1000:  # Simple heuristic
                            best_ls = ls
                    except:
                        pass

        if calls >= max_calls:
            return best_x, best_f

        # Phase 3: Local polish with Nelder-Mead
        if polish_budget > 5:
            try:
                res = minimize(
                    safe_call,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(50, polish_budget), 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
