import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290

        call_count = 0
        best_x = None
        best_f = float('inf')
        all_x = []
        all_f = []

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_c = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x_c))
            all_x.append(x_c.copy())
            all_f.append(f)
            if f < best_f:
                best_f = f
                best_x = x_c.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Phase 0: Initial scatter ---
        n_init = 25
        for i in range(n_init):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            try:
                do_call(x)
            except Exception:
                break

        if best_x is None:
            return (None, float('inf'))

        # --- Gaussian Process surrogate ---
        def gp_predict(xq, X, Y, length_scale, noise):
            """
            GP prediction with RBF kernel.
            X: (n, dim), Y: (n,), xq: (m, dim)
            Returns: mean (m,), variance (m,)
            """
            X = np.atleast_2d(X)
            xq = np.atleast_2d(xq)
            n = X.shape[0]

            # Kernel matrix K(X, X)
            dx = X[:, None, :] - X[None, :, :]
            sqdist = np.sum(dx ** 2, axis=-1)
            K = np.exp(-0.5 * sqdist / (length_scale ** 2)) + noise * np.eye(n)

            # Kernel vector K(xq, X)
            dxq = xq[:, None, :] - X[None, :, :]
            sqdist_q = np.sum(dxq ** 2, axis=-1)
            Ks = np.exp(-0.5 * sqdist_q / (length_scale ** 2))

            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            except np.linalg.LinAlgError:
                try:
                    alpha = np.linalg.solve(K, Y)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.lstsq(K, Y, rcond=None)[0]

            # Mean prediction
            mean = Ks @ alpha

            # Variance prediction
            try:
                v = np.linalg.solve(L, Ks.T)
                var = length_scale ** 2 - np.sum(v ** 2, axis=0)
            except np.linalg.LinAlgError:
                var = np.full(xq.shape[0], length_scale ** 2)

            var = np.maximum(var, 1e-10)
            return mean, var

        def gp_predict_fast(xq, X, Y, length_scale, noise, alpha_cache=None, L_cache=None):
            """
            GP prediction with cached Cholesky factor.
            """
            X = np.atleast_2d(X)
            xq = np.atleast_2d(xq)
            n = X.shape[0]

            if alpha_cache is None or L_cache is None or len(alpha_cache) != n:
                dx = X[:, None, :] - X[None, :, :]
                sqdist = np.sum(dx ** 2, axis=-1)
                K = np.exp(-0.5 * sqdist / (length_scale ** 2)) + noise * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
                except np.linalg.LinAlgError:
                    try:
                        alpha = np.linalg.solve(K, Y)
                    except np.linalg.LinAlgError:
                        alpha = np.linalg.lstsq(K, Y, rcond=None)[0]
                    L = None
            else:
                L = L_cache
                alpha = alpha_cache

            dxq = xq[:, None, :] - X[None, :, :]
            sqdist_q = np.sum(dxq ** 2, axis=-1)
            Ks = np.exp(-0.5 * sqdist_q / (length_scale ** 2))

            mean = Ks @ alpha

            if L is not None:
                try:
                    v = np.linalg.solve(L, Ks.T)
                    var = length_scale ** 2 - np.sum(v ** 2, axis=0)
                except np.linalg.LinAlgError:
                    var = np.full(xq.shape[0], length_scale ** 2)
            else:
                var = np.full(xq.shape[0], length_scale ** 2)

            var = np.maximum(var, 1e-10)
            return mean, var

        def estimate_length_scale(X):
            """Estimate a reasonable length scale from data."""
            n = X.shape[0]
            if n < 2:
                return 1.0
            # Sample pairs
            idx = rng.choice(n, size=min(100, n), replace=False)
            dists = []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    dists.append(np.linalg.norm(X[idx[i]] - X[idx[j]]))
            dists = np.array(dists)
            return max(np.median(dists) * 0.3, 0.1)

        # --- Phase 1: GP-guided optimization ---
        # We'll use a two-stage approach:
        # 1. Use GP with EI acquisition to find promising regions
        # 2. Use local search around the best found point

        n_gp_iters = 30
        gp_length_scale = estimate_length_scale(np.array(all_x))
        gp_noise = 0.01 * (np.max(all_f) - np.min(all_f) + 1.0)

        for it in range(n_gp_iters):
            if call_count >= budget - 20 or time.time() > deadline:
                break

            X_arr = np.array(all_x)
            Y_arr = np.array(all_f)

            # Adaptive length scale: shrink as we converge
            gp_length_scale = max(estimate_length_scale(X_arr) * 0.5, 0.05)

            # Generate candidate points for acquisition
            n_candidates = 200
            candidates = rng.uniform(-4.8, 4.8, (n_candidates, dim))

            # Add points near the current best
            if best_x is not None:
                for _ in range(50):
                    candidates = np.vstack([candidates, best_x + rng.normal(0, gp_length_scale * 0.5, dim)])
            candidates = np.clip(candidates, -4.9, 4.9)

            # GP prediction
            mean, var = gp_predict(candidates, X_arr, Y_arr, gp_length_scale, gp_noise)
            std = np.sqrt(var)

            # Expected Improvement acquisition
            f_min = np.min(Y_arr)
            improvement = f_min - mean
            with np.errstate(divide='ignore', invalid='ignore'):
                z = improvement / (std + 1e-10)
                from scipy.stats import norm
                ei = improvement * norm.cdf(z) + std * norm.pdf(z)
                ei = np.where(std > 1e-8, ei, 0.0)

            # Also consider pure exploration for early iterations
            if it < 5:
                # Mix EI with UCB
                beta = 2.0
                ucb = mean + beta * std
                # Normalize both to [0, 1] range for mixing
                ei_norm = (ei - ei.min()) / (ei.max() - ei.min() + 1e-10)
                ucb_norm = (ucb - ucb.min()) / (ucb.max() - ucb.min() + 1e-10)
                # Lower is better for ucb (we want low function values)
                ucb_norm = 1 - ucb_norm
                score = 0.7 * ei_norm + 0.3 * ucb_norm
            else:
                score = ei

            # Select top candidate
            idx = np.argmax(score)
            x_new = candidates[idx]

            try:
                do_call(x_new)
            except Exception:
                break

        # --- Phase 2: Local refinement with Nelder-Mead ---
        if best_x is not None and call_count < budget - 10 and time.time() < deadline:
            from scipy.optimize import minimize

            # Multiple Nelder-Mead restarts from perturbed best points
            n_nm = 5
            for i in range(n_nm):
                if call_count >= budget - 5 or time.time() > deadline:
                    break

                if i == 0:
                    x0 = best_x.copy()
                else:
                    x0 = best_x + rng.normal(0, 0.5 * (i + 1), dim)
                    x0 = np.clip(x0, -4.9, 4.9)

                # Count function calls for NM
                nm_calls = [0]
                def nm_func(x):
                    nm_calls[0] += 1
                    if nm_calls[0] > 40 or call_count >= budget - 2 or time.time() > deadline:
                        return 1e10
                    return do_call(x)

                try:
                    result = minimize(
                        nm_func, x0, method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -4.9, 4.9)
                except Exception:
                    pass

        # --- Phase 3: Final coordinate descent polish ---
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            step = 0.2
            for _ in range(20):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    for d in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += d * step
                        try:
                            f_try = do_call(x_try)
                        except Exception:
                            break
                        if f_try < f:
                            x[i] = x_try[i]
                            f = f_try
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            best_x = x
            best_f = f

        return best_x, best_f
