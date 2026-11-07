"""Global basin search + Rastrigin-aware local refinement for BBOB f4 (Büche-Rastrigin)."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        call_count = 0
        best_x = np.zeros(dim, dtype=float)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # ---------- Rastrigin-aware local search ----------
        def local_search(x0, f0, max_calls=60, fine=False):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(f0)
            calls_used = 0

            if fine:
                step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            else:
                step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05]

            for step in step_sizes:
                if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                    break
                improved = True
                while improved and calls_used < max_calls and call_count < budget - 5 and time_ok():
                    improved = False
                    for i in range(dim):
                        if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                            break
                        base = x[i]
                        for delta in [step, -step]:
                            if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                                break
                            c = base + delta
                            if not (-4.95 <= c <= 4.95):
                                continue
                            x[i] = c
                            cf = do_call(x)
                            calls_used += 1
                            if cf < f:
                                f = cf
                                improved = True
                            else:
                                x[i] = base
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # ---------- GP Surrogate Model ----------
        class GPSurrogate:
            def __init__(self, dim):
                self.dim = dim
                self.X = None
                self.y = None
                self.alpha = None
                self.K = None
                self.l = np.ones(dim) / 2.0  # length scales
                self.s2 = 1.0  # signal variance
                self.n2 = 1e-8  # noise variance

            def _rbf_kernel(self, X1, X2):
                """RBF kernel with ARD (anisotropic) lengthscales."""
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                # Square and weight by inverse length scales squared
                diff_sq = np.sum(diff ** 2 * (1.0 / self.l ** 2), axis=2)
                return np.exp(-0.5 * diff_sq)

            def fit(self, X, y):
                self.X = np.asarray(X)
                self.y = np.asarray(y)
                n = len(self.y)
                if n < 2:
                    self.alpha = np.zeros(n)
                    return
                # Standardize y for numerical stability
                y_mean = np.mean(self.y)
                y_std = max(np.std(self.y), 1e-8)
                y_std = y_std if y_std > 1e-8 else 1.0
                y_norm = (self.y - y_mean) / y_std

                K = self._rbf_kernel(self.X, self.X) + self.n2 * np.eye(n)
                try:
                    self.alpha = np.linalg.solve(K, y_norm)
                except np.linalg.LinAlgError:
                    self.alpha = np.linalg.lstsq(K, y_norm, rcond=None)[0]
                self.y_mean = y_mean
                self.y_std = y_std

            def predict(self, X_new):
                if self.X is None or len(self.X) < 2:
                    # Return prior mean (0) and high uncertainty
                    return np.zeros(len(X_new)), np.ones(len(X_new))
                K_s = self._rbf_kernel(X_new, self.X)
                K_ss = self._rbf_kernel(X_new, X_new)
                # Predicted mean (in normalized space)
                mu_norm = K_s @ self.alpha
                # Predicted variance
                v = K_s @ self.alpha
                var_norm = np.diag(K_ss) - np.sum(K_s ** 2 * self.alpha, axis=1)
                var_norm = np.maximum(var_norm, 1e-8)
                # Transform back to original scale
                mu = mu_norm * self.y_std + self.y_mean
                var = var_norm * self.y_std ** 2
                return mu, var

            def acquisition(self, X_new, xi=0.01):
                """Expected Improvement acquisition function."""
                mu, var = self.predict(X_new)
                sigma = np.sqrt(var)
                # Avoid division by zero
                sigma_safe = np.maximum(sigma, 1e-8)
                Z = (self.y_min - mu) / sigma_safe
                from scipy.stats import norm
                EI = (self.y_min - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
                return EI

        gp = GPSurrogate(dim)
        gp.y_min = best_f

        def update_gp(X, y):
            gp.X = np.vstack([gp.X, X]) if gp.X is not None else np.asarray(X)
            gp.y = np.concatenate([gp.y, y]) if gp.y is not None else np.asarray(y)
            gp.fit(gp.X, gp.y)
            gp.y_min = np.min(gp.y)

        gp.X = None
        gp.y = None

        def record_point(x, f):
            nonlocal gp
            if gp.X is None:
                gp.X = np.array([x])
                gp.y = np.array([f])
            else:
                gp.X = np.vstack([gp.X, [x]])
                gp.y = np.concatenate([gp.y, [f]])
            gp.fit(gp.X, gp.y)
            gp.y_min = np.min(gp.y)

        # ---------- Phase 0: Initial exploration ----------
        init_pts = [np.zeros(dim)]
        for axis in range(dim):
            for s in (-2.0, -1.0, 1.0, 2.0):
                p = np.zeros(dim)
                p[axis] = s
                init_pts.append(p)
        while len(init_pts) < 20:
            init_pts.append(rng.uniform(-4.0, 4.0, dim))

        for p in init_pts:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.95, 4.95)
            f = do_call(p)
            record_point(p, f)

        # ---------- Phase 1: GP-guided search + local refinement ----------
        reserve = 50
        gp_calls = 0
        max_gp_calls = 200

        while call_count < budget - reserve and time_ok() and gp_calls < max_gp_calls:
            # Use GP to suggest next point
            if gp.X is not None and len(gp.X) >= 5:
                # Generate candidate points
                n_candidates = 200
                candidates = rng.uniform(-4.5, 4.5, size=(n_candidates, dim))
                # Also add some points near current best
                for _ in range(50):
                    candidates = np.vstack([candidates, best_x + rng.normal(0, 0.5, dim)])
                candidates = np.clip(candidates, -4.95, 4.95)

                # Compute acquisition
                ei = gp.acquisition(candidates, xi=0.01)
                # Blend exploitation and exploration
                best_idx = np.argmax(ei)
                next_x = candidates[best_idx]
            else:
                # Not enough data for GP, use random
                next_x = rng.uniform(-4.5, 4.5, dim)
                next_x = np.clip(next_x, -4.95, 4.95)

            f_new = do_call(next_x)
            record_point(next_x, f_new)
            gp_calls += 1

            # Periodically do local refinement
            if gp_calls % 10 == 0 and call_count < budget - reserve - 30 and time_ok():
                local_search(best_x, best_f, max_calls=30, fine=False)

        # ---------- Phase 2: Final fine-tune ----------
        if call_count < budget - 1 and time_ok():
            local_search(best_x, best_f, max_calls=100, fine=True)

        return best_x, best_f
