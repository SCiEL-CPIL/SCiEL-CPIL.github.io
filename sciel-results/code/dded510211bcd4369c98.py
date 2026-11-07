"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free optimizer using Bayesian Optimization with GP surrogate + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.calls = 0

    def _gp_predict(self, X, X_obs, y_obs, length_scale, sigma_f, sigma_n):
        """Gaussian process prediction with Matern kernel."""
        n = X_obs.shape[0]
        m = X.shape[0]
        
        # Matern-5/2 kernel
        def matern_kernel(A, B, ls):
            # A: (n, d), B: (m, d)
            dist = np.sqrt(np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2))
            z = np.sqrt(5.0) * dist / ls
            K = (1 + z + z**2 / 3.0) * np.exp(-z)
            return K
        
        K = matern_kernel(X_obs, X_obs, length_scale) + sigma_n**2 * np.eye(n)
        K_s = matern_kernel(X, X_obs, length_scale)
        
        try:
            L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
        
        mu = K_s @ alpha
        v = np.linalg.solve(L, K_s.T)
        var = sigma_f**2 - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-10)
        
        return mu, var

    def _expected_improvement(self, mu, sigma, best_f, xi=0.01):
        """Compute Expected Improvement."""
        sigma = np.maximum(sigma, 1e-8)
        imp = best_f - mu - xi
        z = imp / sigma
        from scipy.stats import norm
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def __call__(self, func):
        self.calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal best_x, best_f, self
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            self.calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Initial random sampling to build initial dataset
        n_init = max(10, budget // 10)
        X_obs = []
        y_obs = []
        
        for _ in range(n_init):
            if self.calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            val = wrapped_func(x)
            X_obs.append(x)
            y_obs.append(val)
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)

        # Phase 2: Bayesian Optimization with GP surrogate
        # Reserve final 30% of budget for Nelder-Mead refinement
        bo_budget = int(budget * 0.7)
        bo_calls_made = self.calls
        
        # GP hyperparameters
        length_scale = 2.0
        sigma_f = 1.0
        sigma_n = 0.01
        
        # Normalize y for better GP conditioning
        y_min, y_max = y_obs.min(), y_obs.max()
        y_range = y_max - y_min + 1e-8
        y_norm = (y_obs - y_min) / y_range
        
        for it in range(bo_budget - bo_calls_made):
            if self.calls >= budget:
                break
            
            # Generate candidate points
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Predict
            mu, var = self._gp_predict(X_cand, X_obs, y_norm, length_scale, 1.0, sigma_n)
            sigma = np.sqrt(var)
            
            # Expected Improvement
            best_norm = (best_f - y_min) / y_range
            ei = self._expected_improvement(mu, sigma, best_norm, xi=0.01)
            
            # Select point with maximum EI
            idx = np.argmax(ei)
            x_next = X_cand[idx]
            
            # Evaluate
            val = wrapped_func(x_next)
            X_obs = np.vstack([X_obs, x_next])
            y_obs = np.append(y_obs, val)
            
            # Update normalized observations
            y_min, y_max = y_obs.min(), y_obs.max()
            y_range = y_max - y_min + 1e-8
            y_norm = (y_obs - y_min) / y_range
            
            # Adapt length scale based on progress
            if it % 20 == 19:
                length_scale *= 0.8

        # Phase 3: Nelder-Mead refinement from best point
        if self.calls < budget and best_x is not None:
            remaining = budget - self.calls
            if remaining > 10:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': remaining, 'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
