"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with GP surrogate for BBOB-style Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls = 0
        X_obs = []  # list of observed points
        f_obs = []  # list of observed function values

        def safe_eval(x):
            nonlocal calls, best_x, best_fx
            if not np.all(np.isfinite(x)) or x.shape != (n,):
                return np.inf
            if calls >= budget:
                return np.inf
            val = func(x)
            calls += 1
            if np.isfinite(val):
                X_obs.append(x.copy())
                f_obs.append(val)
                if val < best_fx:
                    best_fx = val
                    best_x = x.copy()
            return val

        # Phase 1: Initial Exploration
        # Sample a diverse set of points to build initial GP model
        initial_candidates = [
            np.zeros(n),
            np.ones(n) * 1.0,
            np.ones(n) * 0.5,
            np.ones(n) * -1.0,
            np.ones(n) * -0.5,
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.ones(n) * 3.0,
            np.ones(n) * -3.0
        ]
        
        # Add random points
        n_random = 11
        rand_points = rng.uniform(-4, 4, (n_random, n))
        initial_candidates.extend(rand_points)
        
        for x in initial_candidates:
            if time.time() > deadline or calls >= budget:
                break
            safe_eval(x)

        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)

        if len(X_obs) < 3:
            # Not enough data for GP, just do random search
            while calls < budget and time.time() < deadline:
                x = rng.uniform(-4, 4, n)
                safe_eval(x)
            return best_x, best_fx

        # Phase 2: Bayesian Optimization with GP surrogate
        # Use a simple RBF kernel GP
        def gp_predict(X_new, X_train, y_train, length_scale=1.0, sigma_f=1.0, sigma_n=1e-8):
            """Simple GP prediction with RBF kernel."""
            X_new = np.atleast_2d(X_new)
            X_train = np.atleast_2d(X_train)
            
            # Compute kernel matrix K(X_train, X_train)
            diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            K = sigma_f ** 2 * np.exp(-0.5 * sq_dist / (length_scale ** 2))
            K += sigma_n ** 2 * np.eye(len(X_train))
            
            # Compute K(X_train, X_new)
            diff = X_train[:, np.newaxis, :] - X_new[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            K_s = sigma_f ** 2 * np.exp(-0.5 * sq_dist / (length_scale ** 2))
            
            # Compute K(X_new, X_new)
            diff = X_new[:, np.newaxis, :] - X_new[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            K_ss = sigma_f ** 2 * np.exp(-0.5 * sq_dist / (length_scale ** 2))
            
            # Solve for alpha = K^{-1} y
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
            
            # Mean prediction
            mu = K_s.T @ alpha
            
            # Variance prediction
            v = np.linalg.solve(L, K_s)
            var = np.diag(K_ss) - np.sum(K_s ** 2, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mu, var

        def expected_improvement(mu, var, y_best, xi=0.01):
            """Compute Expected Improvement acquisition function."""
            imp = y_best - mu - xi
            sigma = np.sqrt(var)
            z = imp / sigma
            # Standard normal CDF and PDF
            from scipy.stats import norm
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.where(sigma > 1e-10, ei, 0.0)
            return ei

        def select_next_point(X_train, y_train, y_best, n_candidates=200):
            """Select next point using EI acquisition function."""
            # Generate candidate points in the search space
            candidates = rng.uniform(-4, 4, (n_candidates, n))
            
            # Predict mean and variance at candidates
            mu, var = gp_predict(candidates, X_train, y_train, length_scale=1.0, sigma_f=1.0)
            
            # Compute EI
            ei = expected_improvement(mu, var, y_best, xi=0.01)
            
            # Select candidate with highest EI
            best_idx = np.argmax(ei)
            return candidates[best_idx]

        # BO loop
        while calls < budget and time.time() < deadline:
            # Select next point
            x_next = select_next_point(np.array(X_obs), np.array(f_obs), best_fx)
            
            # Evaluate
            fx = safe_eval(x_next)
            
            # Check for convergence
            if len(f_obs) >= 10:
                recent = f_obs[-10:]
                if np.std(recent) < 1e-6 and np.abs(np.mean(recent) - best_fx) < 1e-6:
                    break

        # Phase 3: Local polish with Nelder-Mead if budget remains
        if best_x is not None and time.time() < deadline and calls < budget - 5:
            def nm_wrapper(x):
                nonlocal calls, best_x, best_fx
                if time.time() > deadline:
                    raise ValueError("Time limit exceeded")
                if calls >= budget:
                    raise ValueError("Budget exceeded")
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls += 1
                if np.isfinite(val) and val < best_fx:
                    best_fx = val
                    best_x = x_clipped.copy()
                return val

            try:
                import scipy.optimize
                res = scipy.optimize.minimize(
                    nm_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(budget - calls, 50),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(n)
            if calls < budget:
                safe_eval(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
