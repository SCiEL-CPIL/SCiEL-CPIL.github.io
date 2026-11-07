"""Trust-region surrogate optimization for RE37 multi-objective task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Trust-region sequential model-based optimization:
    1) Space-filling DOE (Halton + special points).
    2) Fit GP surrogates per objective.
    3) Sequential trust-region optimization: for each weight direction,
       minimize the GP mean within an expanding trust region around the
       current best point.
    4) Adaptive trust-region radius based on improvement.
    5) Exploit remaining budget with targeted local searches.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(0)

        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        # --- GP Surrogate ---
        def matern_kernel(X, Y, length_scale=1.0, variance=1.0, nu=2.5):
            sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
            sq_dists = np.maximum(sq_dists, 0)
            dist = np.sqrt(sq_dists)
            kernel = np.sqrt(3) * dist / length_scale
            return variance * (1 + kernel) * np.exp(-kernel)

        class GPSurrogate:
            def __init__(self, X, Y, length_scale=0.4):
                self.X = X
                self.Y = Y
                self.mu = np.mean(Y, axis=0)
                self.sigma = np.std(Y, axis=0) + 1e-10
                self.Y_scaled = (Y - self.mu) / self.sigma
                self.length_scale = length_scale
                self.variance = 1.0
                self.K = matern_kernel(X, X, length_scale, self.variance) + 1e-6 * np.eye(X.shape[0])
                try:
                    self.alpha = np.linalg.solve(self.K, self.Y_scaled)
                except np.linalg.LinAlgError:
                    self.K += 1e-4 * np.eye(X.shape[0])
                    self.alpha = np.linalg.solve(self.K, self.Y_scaled)

            def predict(self, X_new):
                K_s = matern_kernel(X_new, self.X, self.length_scale, self.variance)
                means = self.mu + K_s @ self.alpha * self.sigma
                v = matern_kernel(X_new, X_new, self.length_scale, self.variance)
                v = np.diag(v) - np.sum(K_s @ self.K @ K_s.T, axis=1)
                v = np.maximum(v, 1e-10)
                stds = np.sqrt(v) * self.sigma
                return means, stds

        # --- 1) DOE: Halton sequence + special points ---
        def halton_sequence(n, dim):
            def halton_one(i, base):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                for i in range(1, n + 1):
                    points[i-1, d] = halton_one(i, base)
            return points

        n_halton = 35
        halton_pts = halton_sequence(n_halton, dim)

        specials = [lo, hi, (lo+hi)/2.0,
                    np.array([0,0,1,1]), np.array([0,1,0,1]),
                    np.array([0,1,1,0]), np.array([1,0,0,1]),
                    np.array([1,0,1,0]), np.array([1,1,0,0])]
        specials = np.array([np.clip(s, lo, hi) for s in specials], dtype=float)

        grid = np.vstack([specials, halton_pts])
        rng.shuffle(grid)

        model_X = []
        model_Y = []
        pareto = {}

        def add_point(x, f):
            key = tuple(np.round(np.asarray(f, dtype=float), 12))
            if key not in pareto:
                pareto[key] = np.asarray(x, dtype=float).copy()

        for x in grid:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            try:
                f = eval_real(x)
                model_X.append(np.asarray(x, dtype=float))
                model_Y.append(np.asarray(f, dtype=float))
                add_point(x, f)
            except Exception:
                break

        if len(model_X) == 0:
            return None

        Xtr = np.vstack(model_X)
        Ytr = np.vstack(model_Y)

        def fit_gps():
            gps = []
            for k in range(n_obj):
                try:
                    gps.append(GPSurrogate(Xtr, Ytr[:, k], length_scale=0.4))
                except Exception:
                    gps.append(None)
            return gps

        def surrogate_predict(gps, X_new):
            if all(g is None for g in gps):
                return np.tile(np.mean(Ytr, axis=0), (X_new.shape[0], 1))
            means = np.zeros((X_new.shape[0], n_obj))
            for k, g in enumerate(gps):
                if g is not None:
                    m, _ = g.predict(X_new)
                    means[:, k] = m
                else:
                    means[:, k] = np.mean(Ytr[:, k])
            return means

        # --- 2) Trust-region sequential optimization ---
        # Generate diverse weight directions
        weights = []
        for k in range(n_obj):
            w = np.zeros(n_obj); w[k] = 1.0; weights.append(w)
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                w = np.zeros(n_obj); w[i] = 0.5; w[j] = 0.5; weights.append(w)
        for _ in range(15):
            w = rng.dirichlet([1.0, 1.0, 1.0])
            weights.append(w)

        # Find initial best points per weight from existing data
        gps = fit_gps()

        def trust_region_optimize(w, center, radius, max_iters=12):
            """Minimize weighted GP mean within trust region around center."""
            best_x = center.copy()
            best_f = None
            best_scalar = np.inf
            cur_radius = radius

            for it in range(max_iters):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break

                # Sample candidates within trust region
                n_cand = 30
                cand = center + rng.normal(0, cur_radius, size=(n_cand, dim))
                cand = np.clip(cand, lo, hi)
                # Also add center and a few axis-aligned perturbations
                axis_cands = []
                for d in range(dim):
                    for sign in [-1, 1]:
                        xc = center.copy()
                        xc[d] += sign * cur_radius * 0.5
                        axis_cands.append(np.clip(xc, lo, hi))
                cand = np.vstack([cand, np.array(axis_cands)])

                # Predict with surrogate
                Yc = surrogate_predict(gps, cand)
                scalars = Yc @ w
                # Select top candidates to evaluate
                top_idx = np.argsort(scalars)[:5]
                top_cands = cand[top_idx]

                best_cand_x = None
                best_cand_f = None
                best_cand_s = np.inf

                for xc in top_cands:
                    if self.eval_count >= self.budget or time.time() > deadline:
                        break
                    try:
                        fc = eval_real(xc)
                    except Exception:
                        break
                    sc = fc @ w
                    if sc < best_cand_s:
                        best_cand_s = sc
                        best_cand_x = xc.copy()
                        best_cand_f = fc.copy()

                if best_cand_x is None:
                    break

                # Update best
                if best_cand_s < best_scalar:
                    improvement = best_scalar - best_cand_s
                    best_scalar = best_cand_s
                    best_x = best_cand_x
                    best_f = best_cand_f
                    # Shrink trust region on improvement
                    cur_radius = max(cur_radius * 0.7, 0.02)
                else:
                    # Expand trust region on no improvement
                    cur_radius = min(cur_radius * 1.4, 0.5)

                # Update model
                model_X.append(best_cand_x)
                model_Y.append(best_cand_f)
                Xtr = np.vstack(model_X)
                Ytr = np.vstack(model_Y)
                gps = fit_gps()

                add_point(best_cand_x, best_cand_f)

            return best_x, best_f

        # Determine initial centers: best points from DOE
        for w in weights:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            # Find best existing point for this weight
            scalars = Ytr @ w
            best_idx = np.argmin(scalars)
            center = Xtr[best_idx].copy()
            radius = 0.3

            bx, bf = trust_region_optimize(w, center, radius, max_iters=10)
            if bf is not None:
                add_point(bx, bf)

        # --- 3) Exploit remaining budget with more trust-region iterations ---
        # Re-fit with all data
        Xtr = np.vstack(model_X)
        Ytr = np.vstack(model_Y)
        gps = fit_gps()

        # Additional weight directions for finer coverage
        extra_weights = [rng.dirichlet([1.0, 1.0, 1.0]) for _ in range(20)]

        for w in extra_weights:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            scalars = Ytr @ w
            best_idx = np.argmin(scalars)
            center = Xtr[best_idx].copy()
            radius = 0.15

            bx, bf = trust_region_optimize(w, center, radius, max_iters=8)
            if bf is not None:
                add_point(bx, bf)

        # --- 4) Final exploitation: local search around Pareto points ---
        def local_search(x0, max_evals=5):
            """Simple coordinate-descent local search."""
            best_x = np.asarray(x0, dtype=float).copy()
            best_f = eval_real(best_x)
            steps = 0.05
            for _ in range(max_evals):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if self.eval_count >= self.budget or time.time() > deadline:
                        break
                    for sign in [-1, 1]:
                        xc = best_x.copy()
                        xc[d] += sign * steps
                        xc = np.clip(xc, lo, hi)
                        try:
                            fc = eval_real(xc)
                        except Exception:
                            continue
                        if np.any(fc < best_f) and not np.all(fc > best_f):
                            best_x = xc
                            best_f = fc
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    steps *= 0.5
                    if steps < 0.01:
                        break
            return best_x, best_f

        # Local search around Pareto-optimal points
        pareto_pts = []
        for f, x in pareto.items():
            pareto_pts.append((x, np.asarray(f, dtype=float)))

        # Filter to actual Pareto-optimal points
        if len(pareto_pts) > 0:
            pareto_xs = np.array([p[0] for p in pareto_pts])
            pareto_fs = np.array([p[1] for p in pareto_pts])
            # Keep only non-dominated
            n_pts = len(pareto_fs)
            dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i != j and np.all(pareto_fs[j] <= pareto_fs[i]) and np.any(pareto_fs[j] < pareto_fs[i]):
                        dominated[i] = True
                        break
            nondom_idx = np.where(~dominated)[0]

            # Local search around a subset of non-dominated points
            max_ls = min(10, len(nondom_idx))
            rng_idx = rng.choice(len(nondom_idx), size=max_ls, replace=False)
            for idx in rng_idx:
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                x0 = pareto_xs[nondom_idx[idx]]
                bx, bf = local_search(x0, max_evals=4)
                add_point(bx, bf)

        return None
# EVOLVE-BLOCK-END
