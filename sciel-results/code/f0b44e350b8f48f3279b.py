"""Multi-Fidelity MBO with GP surrogate and EI acquisition for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

class Optimizer:
    """GP-based MBO optimizer for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        total_budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---- GP helper ----
        def gp_ei_candidates(X, Y, n_cand, xi, batch, rng):
            """Return n_cand candidate points chosen by greedy EI with decorrelation."""
            n = X.shape[0]
            # Standardize for kernel
            mu_x = X.mean(axis=0)
            sd_x = X.std(axis=0)
            sd_x[sd_x < 1e-10] = 1.0
            Xs = (X - mu_x) / sd_x

            # Hyperparameters (fixed for speed)
            ell = 1.5
            var_y = max(Y.std() ** 2, 1e-6)
            noise = 1e-6 * var_y

            K = np.exp(-0.5 * cdist(Xs, Xs) ** 2 / ell ** 2)
            K[np.diag_indices(n)] += noise

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K[np.diag_indices(n)] += 1e-4 * var_y
                L = np.linalg.cholesky(K)

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y - Y.mean()))
            f_mean_const = Y.mean()

            selected = []
            # Greedy EI with decorrelation
            for _ in range(n_cand):
                # Generate random candidates
                n_rand = 2000
                Xc = rng.uniform(-5, 5, (n_rand, dim))
                # Resample out-of-bound (all are in bound here)

                Xcs = (Xc - mu_x) / sd_x
                Kc = np.exp(-0.5 * cdist(Xcs, Xs) ** 2 / ell ** 2)

                # GP mean and variance
                f_mean = f_mean_const + Kc @ alpha
                v = np.linalg.solve(L, Kc.T)
                f_var = var_y - np.sum(v ** 2, axis=0)
                f_var = np.maximum(f_var, 1e-12)
                f_std = np.sqrt(f_var)

                # EI
                z = (best_f - xi - f_mean) / f_std
                from scipy.stats import norm
                ei = f_std * (z * norm.pdf(z) + norm.cdf(z))

                # Decorrelation penalty: penalize points close to already-selected batch
                if selected:
                    sel_arr = np.array(selected)
                    dists = cdist(Xc, sel_arr)
                    min_dists = dists.min(axis=1)
                    # Penalize if too close (within 0.5 in any dim)
                    ei = ei - 2.0 * np.exp(-min_dists ** 2 / (2 * 0.3 ** 2))

                best_idx = np.argmax(ei)
                selected.append(Xc[best_idx])

            return np.array(selected)

        # ---- Phase 1: Initial exploration ----
        n_explore = min(25, total_budget // 10)
        # Latin hypercube for better coverage
        rng = np.random.default_rng(42)
        for d in range(dim):
            pass
        # Simple LHS
        lhs_points = np.zeros((n_explore, dim))
        for d in range(dim):
            perm = rng.permutation(n_explore)
            lhs_points[:, d] = (perm + 0.5) / n_explore * 10.0 - 5.0
        for i in range(n_explore):
            if time.time() > deadline or calls_used >= total_budget:
                break
            x = lhs_points[i]
            fx = func(x)
            track_call(x, fx)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        # ---- Phase 2: GP-MBO loop ----
        X = np.array([best_x] + [lhs_points[i] for i in range(n_explore) if calls_used > i + 1])
        # Rebuild X, Y from tracked calls
        X = np.array([lhs_points[i] for i in range(n_explore)])
        # We need to store all evaluated points. Let's redo:
        # Actually, let's collect during exploration
        # Redo: we'll just use a list

        # Reset and redo properly
        calls_used = 0
        best_x = None
        best_f = np.inf
        eval_points = []
        eval_values = []

        def track_call2(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            eval_points.append(x.copy())
            eval_values.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Latin hypercube
        for i in range(n_explore):
            if time.time() > deadline or calls_used >= total_budget:
                break
            x = lhs_points[i]
            fx = func(x)
            track_call2(x, fx)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call2(best_x, best_f)

        # Phase 2: GP-MBO
        stagnation_count = 0
        batch_size = 5
        xi_base = 0.01
        prev_best = best_f

        while calls_used + batch_size <= total_budget and time.time() < deadline:
            # Check time for GP fit
            if time.time() > deadline - 5.0:
                break

            X = np.array(eval_points)
            Y = np.array(eval_values)
            n = X.shape[0]

            if n < 3:
                # Not enough data, do uniform
                for _ in range(batch_size):
                    if time.time() > deadline or calls_used >= total_budget:
                        break
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    track_call2(x, fx)
                continue

            # Adaptive xi
            if best_f < prev_best - 1e-6:
                stagnation_count = 0
            else:
                stagnation_count += 1
            prev_best = best_f
            xi = xi_base * (1.0 + 0.5 * stagnation_count)

            # Get candidates from GP
            try:
                candidates = gp_ei_candidates(X, Y, batch_size, xi, np.array(eval_points), rng)
            except Exception:
                candidates = rng.uniform(-5, 5, (batch_size, dim))

            # Evaluate candidates
            for i in range(batch_size):
                if time.time() > deadline or calls_used >= total_budget:
                    break
                x = candidates[i]
                # Resample if out of bounds (shouldn't happen but safe)
                if np.any(x < -5) or np.any(x > 5):
                    x = rng.uniform(-5, 5, dim)
                fx = func(x)
                track_call2(x, fx)

        return best_x, best_f
# EVOLVE-BLOCK-END
