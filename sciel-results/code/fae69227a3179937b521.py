"""Bayesian Optimization with GP surrogate for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP-based Bayesian Optimization with local refinement for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Safe call wrapper
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                if x.shape != (dim,):
                    x = x.reshape(dim)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                if not np.isfinite(f):
                    f = 1e10
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        # --- Phase 1: Initial sampling to build GP training set ---
        rng = np.random.default_rng(42)
        X = []
        Y = []

        # Sample initial points: mix of random, grid-like, and center
        n_init = min(30, budget)
        # Center
        center = np.zeros(dim)
        f_center = safe_call(center)
        if f_center is not None:
            X.append(center)
            Y.append(f_center)

        # Random points
        for i in range(n_init - 1):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_call(x)
            if f is not None:
                X.append(x)
                Y.append(f)

        X = np.array(X)
        Y = np.array(Y)

        # --- GP surrogate using RBF kernel ---
        def gp_predict(X_train, Y_train, X_test, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6):
            """Simple GP with RBF kernel. Returns mean and variance."""
            n = len(X_train)
            d = X_train.shape[1]
            # Compute kernel matrix
            diff = X_train[:, None, :] - X_train[None, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            K = sigma_f ** 2 * np.exp(-sq_dist / (2 * length_scale ** 2)) + sigma_n ** 2 * np.eye(n)
            K += 1e-8 * np.eye(n)  # numerical stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-4 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Fallback: return mean of training data
                    mu = np.mean(Y_train)
                    return np.full(len(X_test), mu), np.full(len(X_test), np.var(Y_train) + 1.0)

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train - np.mean(Y_train)))

            # Predict at test points
            diff_test = X_test[:, None, :] - X_train[None, :, :]
            sq_dist_test = np.sum(diff_test ** 2, axis=2)
            k_star = sigma_f ** 2 * np.exp(-sq_dist_test / (2 * length_scale ** 2))

            mu = k_star @ alpha + np.mean(Y_train)
            v = np.linalg.solve(L, k_star.T)
            var = sigma_f ** 2 - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-8)
            return mu, var

        def expected_improvement(mu, var, best_y, xi=0.01):
            """Expected improvement acquisition function."""
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            z = (best_y - mu - xi) / sigma
            # Avoid division by zero
            z = np.where(sigma < 1e-10, 0.0, z)
            ei = (best_y - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # --- Phase 2: Bayesian Optimization loop ---
        # Use a subset of budget for BO, rest for local refinement
        bo_budget = int(budget * 0.6)
        local_budget = budget - bo_budget

        current_bo_calls = 0
        max_bo_iters = bo_budget

        # Adaptive length scale: start with domain scale
        length_scale = 2.0
        sigma_f = 1.0

        for bo_iter in range(max_bo_iters):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Update GP parameters based on observed spread
            if len(Y) >= 2:
                sigma_f = max(np.std(Y), 1e-3)
                # Estimate length scale from typical spacing
                if len(X) >= 2:
                    diffs = np.linalg.norm(X[0] - X[1:])
                    length_scale = max(np.median(diffs) * 0.5, 0.5)

            # Generate candidate points for acquisition
            n_candidates = 200
            # Mix of random and structured candidates
            candidates = []
            # Random
            cand_rand = rng.uniform(-5, 5, (n_candidates // 2, dim))
            candidates.append(cand_rand)
            # Around current best
            if best_x is not None:
                cand_local = best_x + rng.normal(0, 0.5, (n_candidates // 4, dim))
                candidates.append(cand_local)
                cand_wider = best_x + rng.normal(0, 1.5, (n_candidates // 4, dim))
                candidates.append(cand_wider)

            X_cand = np.vstack(candidates)
            X_cand = np.clip(X_cand, -5, 5)

            # GP prediction
            mu, var = gp_predict(X, Y, X_cand, length_scale, sigma_f)
            best_y = np.min(Y)
            ei = expected_improvement(mu, var, best_y, xi=0.01)

            # Select top candidates with diversity
            top_k = min(10, len(ei))
            top_indices = np.argsort(ei)[::-1][:top_k]

            # Pick the best EI point
            x_new = X_cand[top_indices[0]]

            f_new = safe_call(x_new)
            if f_new is None:
                continue

            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)

            # Occasionally do a local refinement around best
            if bo_iter % 5 == 4 and best_x is not None and calls_made < budget - 10:
                # Quick local polish
                try:
                    res = minimize(lambda x: safe_call(x) if safe_call(x) is not None else 1e10,
                                   best_x, method='Nelder-Mead',
                                   options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_x = res.x.copy()
                        best_f = res.fun
                except Exception:
                    pass

        # --- Phase 3: Local refinement with remaining budget ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Multiple local refinement attempts
            n_refine = min(3, (budget - calls_made) // 20)
            for r in range(n_refine):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 5:
                    break
                try:
                    # Perturb slightly for diversity in refinement
                    if r == 0:
                        x_start = best_x.copy()
                    else:
                        x_start = best_x + rng.normal(0, 0.1, dim)
                    x_start = np.clip(x_start, -5, 5)

                    res = minimize(lambda x: safe_call(x) if safe_call(x) is not None else 1e10,
                                   x_start, method='Nelder-Mead',
                                   options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_x = np.clip(res.x, -5, 5)
                        best_f = res.fun
                except Exception:
                    pass

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            if best_f is None:
                best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
