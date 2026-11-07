"""Rewritten optimizer using Bayesian Optimization with GP surrogate for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Bayesian Optimization with GP surrogate for Büche-Rastrigin (f4)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        dim = self.dim

        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling using Latin Hypercube for good coverage
        # Büche-Rastrigin has periodic structure; use enough points to find basins
        n_init = min(80, max_calls // 5)
        X_samples = np.random.uniform(-4.5, 4.5, size=(n_init, dim))
        # Add some structured points near origin and periodic locations
        structured = np.array([
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.concatenate([np.zeros(dim//2), np.full(dim - dim//2, 3.0)]),
            np.concatenate([np.full(dim//2, -3.0), np.zeros(dim - dim//2)]),
        ])
        # Combine and evaluate
        all_init = np.vstack([X_samples, structured[:min(5, max_calls - n_init)]])
        all_init = all_init[:min(n_init + 5, max_calls)]
        
        X_observed = []
        Y_observed = []
        for x in all_init:
            if call_count >= max_calls or time.time() > deadline:
                break
            f = counted_func(x)
            X_observed.append(x)
            Y_observed.append(f)

        # Phase 2: Bayesian Optimization loop
        # Use GP surrogate with LCB acquisition
        from scipy.spatial.distance import cdist

        def gp_predict(X_new, X_obs, y_obs, length_scale=1.0, sigma=1.0, noise=1e-6):
            """Simple GP prediction using RBF kernel."""
            n = len(X_obs)
            # Compute kernel matrix
            D = cdist(X_obs, X_obs)
            K = sigma**2 * np.exp(-0.5 * D**2 / (length_scale**2))
            K[np.diag_indices_from(K)] += noise

            # Cholesky for stability
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n))

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))

            # Predict at X_new
            K_new = sigma**2 * np.exp(-0.5 * cdist(X_new, X_obs)**2 / (length_scale**2))
            mu = K_new @ alpha
            v = np.linalg.solve(L, K_new.T)
            var = sigma**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        def lcb_acquisition(X_new, X_obs, y_obs, beta=2.0, length_scale=1.0, sigma=1.0):
            """Lower Confidence Bound acquisition."""
            mu, sigma_pred = gp_predict(X_new, X_obs, y_obs, length_scale, sigma)
            return mu - beta * sigma_pred

        # Adaptive length scale and sigma based on observed data
        def estimate_hyperparams(X_obs, y_obs):
            """Simple heuristic for GP hyperparameters."""
            if len(X_obs) < 2:
                return 1.0, 1.0
            # Estimate length scale from nearest neighbor distances
            D = cdist(X_obs, X_obs)
            np.fill_diagonal(D, np.inf)
            nn_dists = np.min(D, axis=1)
            length_scale = np.median(nn_dists) * 0.5
            length_scale = np.clip(length_scale, 0.1, 5.0)
            # Estimate sigma from y range
            y_range = np.ptp(y_obs)
            sigma = max(y_range / 10.0, 1.0)
            return length_scale, sigma

        # BO loop
        bo_iterations = 0
        while call_count < max_calls and time.time() < deadline:
            bo_iterations += 1

            # Estimate hyperparameters
            if len(X_observed) >= 2:
                length_scale, sigma = estimate_hyperparams(np.array(X_observed), np.array(Y_observed))
            else:
                length_scale, sigma = 1.0, 1.0

            # Generate candidate points for acquisition
            # Mix of random and grid-based candidates
            n_candidates = min(200, (max_calls - call_count) * 2)
            X_cand = np.random.uniform(-5, 5, size=(n_candidates, dim))
            
            # Add some structured candidates
            extra_cands = []
            for _ in range(min(10, n_candidates // 20)):
                base = np.random.choice(X_observed) if X_observed else np.zeros(dim)
                perturb = np.random.normal(0, 0.5, dim)
                extra_cands.append(np.clip(base + perturb, -5, 5))
            if extra_cands:
                X_cand = np.vstack([X_cand, np.array(extra_cands)])

            # Compute acquisition
            if len(X_observed) >= 2:
                acq_vals = lcb_acquisition(X_cand, np.array(X_observed), np.array(Y_observed),
                                           beta=2.0, length_scale=length_scale, sigma=sigma)
                # Select top candidates
                n_select = min(5, len(acq_vals))
                top_indices = np.argsort(acq_vals)[:n_select]
            else:
                top_indices = np.random.choice(len(X_cand), min(5, len(X_cand)), replace=False)

            # Evaluate selected candidates
            for idx in top_indices:
                if call_count >= max_calls or time.time() > deadline:
                    break
                x_new = X_cand[idx]
                f_new = counted_func(x_new)
                X_observed.append(x_new)
                Y_observed.append(f_new)

        # Phase 3: Final local polish with Nelder-Mead
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining >= 20:
                try:
                    def nm_func(x):
                        return counted_func(np.clip(x, -5, 5))
                    
                    max_iter = max(5, remaining // (dim + 1))
                    result = minimize(
                        nm_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-5,
                            'fatol': 1e-5,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
