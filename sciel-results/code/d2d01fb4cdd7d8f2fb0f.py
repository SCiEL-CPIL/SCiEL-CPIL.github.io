# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

class Optimizer:
    """Bayesian Optimization with GP surrogate for Bent Cigar (f12).
    
    Uses a Gaussian Process with Matern-5/2 kernel to model the landscape,
    and Expected Improvement acquisition to guide sampling toward the
    narrow anisotropic valley without relying on gradient information.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                val = float(func(x_clipped))
                if np.isfinite(val) and val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)

        # --- Initial diverse sampling ---
        n_init = min(12, budget - 5)
        X_init = np.empty((n_init, dim))
        X_init[0] = np.zeros(dim)
        for i in range(1, n_init):
            X_init[i] = rng.uniform(-5.0, 5.0, dim)

        Y = np.empty(n_init)
        for i in range(n_init):
            Y[i] = wrapper(X_init[i])
            if not np.isfinite(Y[i]):
                Y[i] = 1e10

        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # --- GP Kernel: Matern 5/2 with ARD ---
        def matern52_kernel(X1, X2, lengthscales, variance, noise):
            d = np.linalg.norm(X1[:, np.newaxis, :] - X2[np.newaxis, :, :], axis=2)
            sd = d / lengthscales[np.newaxis, np.newaxis, :]
            K = variance * (1.0 + np.sqrt(5.0) * sd + 5.0 / 3.0 * sd ** 2) * np.exp(-np.sqrt(5.0) * sd)
            K = np.sum(K, axis=2)
            if X1.shape[0] == X2.shape[0] and X1.shape[0] == X2.shape[0]:
                K[np.arange(K.shape[0]), np.arange(K.shape[0])] += noise
            return K

        def gp_predict(X_new, X_obs, Y_obs, lengthscales, variance, noise):
            n = X_obs.shape[0]
            K = matern52_kernel(X_obs, X_obs, lengthscales, variance, noise)
            K += 1e-10 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            except np.linalg.LinAlgError:
                return np.full(X_new.shape[0], np.mean(Y_obs)), np.full(X_new.shape[0], variance)
            
            K_s = matern52_kernel(X_new, X_obs, lengthscales, variance, 0.0)
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            K_ss = variance * np.ones((X_new.shape[0],))
            var = np.maximum(K_ss - np.sum(v ** 2, axis=1), 1e-12)
            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, f_best, xi=0.01):
            eps = 1e-10
            sigma = np.maximum(sigma, eps)
            z = (f_best - mu - xi) / sigma
            from scipy.stats import norm
            ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        def maximize_acquisition(X_obs, Y_obs, lengthscales, variance, noise, f_best, n_candidates=2000):
            # Generate candidate points: mix of random and perturbed around best
            candidates = np.empty((n_candidates, dim))
            n_rand = int(n_candidates * 0.6)
            n_perturb = n_candidates - n_rand
            
            candidates[:n_rand] = rng.uniform(-5.0, 5.0, (n_rand, dim))
            
            # Perturb around best point and other good points
            good_indices = np.argsort(Y_obs)[:min(5, len(Y_obs))]
            for i in range(n_perturb):
                base = X_obs[good_indices[i % len(good_indices)]]
                scale = 0.5 + 0.3 * (i / max(1, n_perturb))
                candidates[n_rand + i] = np.clip(base + scale * rng.standard_normal(dim), -5.0, 5.0)
            
            mu, sigma = gp_predict(candidates, X_obs, Y_obs, lengthscales, variance, noise)
            ei = expected_improvement(mu, sigma, f_best)
            best_idx = np.argmax(ei)
            return candidates[best_idx]

        # --- Main BO loop ---
        # Initial kernel parameters
        lengthscales = np.ones(dim) * 2.0
        variance = 1.0
        noise = 1e-6

        # Scale Y for GP stability
        Y_mean = np.mean(Y)
        Y_std = max(np.std(Y), 1e-6)
        Y_scaled = (Y - Y_mean) / Y_std

        # Fit initial kernel params roughly
        def neg_log_likelihood(params):
            if time.time() > deadline:
                return 1e10
            ls = np.exp(params[:dim])
            var = np.exp(params[dim])
            noi = np.exp(params[dim + 1])
            n = X_obs.shape[0]
            K = matern52_kernel(X_obs, X_obs, ls, var, noi)
            K += 1e-8 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_scaled))
                nll = 0.5 * Y_scaled @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            except np.linalg.LinAlgError:
                return 1e10
            return nll

        X_obs = X_init.copy()
        Y_obs = Y.copy()
        n_obs = n_init

        # Fit kernel parameters (limited iterations for speed)
        try:
            x0 = np.concatenate([np.log(lengthscales), np.log(variance), np.log(noise)])
            res_fit = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                              bounds=[(np.log(0.1), np.log(20.0))] * dim + [(np.log(0.01), np.log(100.0)), (np.log(1e-8), np.log(1.0))],
                              options={'maxiter': 30, 'maxfev': 50})
            if res_fit.fun < 1e10:
                lengthscales = np.exp(res_fit.x[:dim])
                variance = np.exp(res_fit.x[dim])
                noise = np.exp(res_fit.x[dim + 1])
        except Exception:
            pass

        # BO loop
        n_bo = budget - calls - 5
        if n_bo < 5:
            return best_x, best_f

        for it in range(n_bo):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            f_best = np.min(Y_obs)

            # Get next point from acquisition
            x_next = maximize_acquisition(X_obs, Y_obs, lengthscales, variance, noise, f_best, n_candidates=1500)

            f_next = wrapper(x_next)
            if not np.isfinite(f_next):
                f_next = f_best + 1.0

            # Add to observation set
            n_obs += 1
            X_obs = np.vstack([X_obs, x_next])
            Y_obs = np.append(Y_obs, f_next)

            # Update Y scaling periodically
            if it % 10 == 0 and n_obs > 10:
                Y_mean = np.mean(Y_obs)
                Y_std = max(np.std(Y_obs), 1e-6)

            # Refit kernel every 15 iterations
            if it > 0 and it % 15 == 0 and n_obs > 15:
                Y_scaled = (Y_obs - Y_mean) / Y_std
                X_obs_refit = X_obs
                try:
                    x0 = np.concatenate([np.log(lengthscales), np.log(variance), np.log(noise)])
                    res_fit = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                                      bounds=[(np.log(0.1), np.log(20.0))] * dim + [(np.log(0.01), np.log(100.0)), (np.log(1e-8), np.log(1.0))],
                                      options={'maxiter': 20, 'maxfev': 30})
                    if res_fit.fun < 1e10:
                        lengthscales = np.exp(res_fit.x[:dim])
                        variance = np.exp(res_fit.x[dim])
                        noise = np.exp(res_fit.x[dim + 1])
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
