"""Model-and-optimize for RE37 multi-objective task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-and-optimize with Gaussian Process Surrogate:
    1) Deterministic DOE to sample objectives.
    2) Fit a GP surrogate per objective.
    3) Propose points by minimizing weighted sum of GP means (and exploring via variance if needed, but mean is usually sufficient for optimization).
    4) Local search on real problem.
    5) Interpolation and remaining budget for random restarts.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0
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

        # --- GP Surrogate Implementation ---
        def matern_kernel(X, Y, length_scale=1.0, variance=1.0, nu=2.5):
            # X: (n, d), Y: (m, d)
            sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
            sq_dists = np.maximum(sq_dists, 0)
            dist = np.sqrt(sq_dists)
            if nu == 2.5:
                # Matern 2.5
                kernel = np.sqrt(3) * dist / length_scale
                return variance * (1 + kernel) * np.exp(-kernel)
            elif nu == 5.0:
                # Matern 5.0 (close to RBF)
                kernel = np.sqrt(5) * dist / length_scale
                return variance * (1 + kernel + kernel**2/3) * np.exp(-kernel)
            else:
                # RBF
                return variance * np.exp(-0.5 * dist**2 / length_scale**2)

        class GPSurrogate:
            def __init__(self, X, Y, length_scale=0.5):
                self.X = X
                self.Y = Y
                self.mu = np.mean(Y, axis=0)
                self.sigma = np.std(Y, axis=0)
                self.Y_scaled = (Y - self.mu) / (self.sigma + 1e-10)
                self.length_scale = length_scale
                self.variance = 1.0
                self.K = matern_kernel(X, X, length_scale, self.variance) + 1e-6 * np.eye(X.shape[0])
                try:
                    self.alpha = np.linalg.solve(self.K, self.Y_scaled)
                except np.linalg.LinAlgError:
                    # Regularize if singular
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

        # --- 1) DOE ---
        levels = np.linspace(0.0, 1.0, 4)
        grid = np.array(np.meshgrid(*([levels]*dim)), dtype=float)
        grid = grid.reshape(dim, -1).T
        grid = grid[::2]
        rng.shuffle(grid)
        
        specials = [lo, hi, (lo+hi)/2.0,
                    np.array([0,0,1,1]), np.array([0,1,0,1]),
                    np.array([0,1,1,0]), np.array([1,0,0,1]),
                    np.array([1,0,1,0]), np.array([1,1,0,0])]
        specials = np.array([np.clip(s, lo, hi) for s in specials], dtype=float)
        grid = np.vstack([specials, grid])

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

        # Fit GPs
        gps = []
        for k in range(n_obj):
            try:
                gps.append(GPSurrogate(Xtr, Ytr[:, k], length_scale=0.5))
            except Exception:
                # Fallback to linear if GP fails
                gps.append(None)

        def surrogate_predict(X_new):
            if all(g is None for g in gps):
                # Fallback: linear interpolation or just return mean
                return np.tile(np.mean(Ytr, axis=0), (X_new.shape[0], 1))
            
            means = np.zeros((X_new.shape[0], n_obj))
            for k, g in enumerate(gps):
                if g is not None:
                    m, _ = g.predict(X_new)
                    means[:, k] = m
                else:
                    means[:, k] = np.mean(Ytr[:, k])
            return means

        # --- 2) Weighted Sum Surrogate Search ---
        weights = []
        for k in range(n_obj):
            w = np.zeros(n_obj); w[k] = 1.0; weights.append(w)
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                w = np.zeros(n_obj); w[i] = 0.5; w[j] = 0.5; weights.append(w)
        for _ in range(10):
            w = rng.dirichlet([1.0, 1.0, 1.0]); weights.append(w)

        proposals = []
        # Sample candidates
        cand = rng.uniform(lo, hi, size=(500, dim))
        Yc = surrogate_predict(cand)
        
        for w in weights:
            s = Yc @ w
            idx = np.argsort(s)[:10]
            for i in idx:
                proposals.append(cand[i])

        # Also include best from training set
        for w in weights:
            s = Ytr @ w
            idx = np.argsort(s)[:5]
            for i in idx:
                proposals.append(Xtr[i])

        # Deduplicate
        seen = set()
        unique_prop = []
        for x in proposals:
            k = tuple(np.round(np.clip(x, lo, hi), 6))
            if k not in seen:
                seen.add(k)
                unique_prop.append(np.clip(x, lo, hi))
        rng.shuffle(unique_prop)

        # --- 3) Local Real Optimization ---
        def local_hill(x, steps=20):
            best = np.asarray(x, dtype=float).copy()
            bestf = eval_real(best)
            for i in range(steps):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                radius = 0.05 + 0.10 * (1.0 - i / max(steps, 1))
                step = rng.normal(0.0, radius, size=dim)
                cand = np.clip(best + step, lo, hi)
                f = eval_real(cand)
                if np.all(f <= bestf) and np.any(f < bestf):
                    best = cand
                    bestf = f
            return best, bestf

        for x in unique_prop:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            b, bf = local_hill(x, steps=15)
            add_point(b, bf)

        # --- 4) Interpolation ---
        def pareto_points():
            pts = []
            for f, x in pareto.items():
                pts.append((x, np.asarray(f, dtype=float)))
            return pts

        used = set()
        for _ in range(30):
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            pts = pareto_points()
            if len(pts) < 2:
                break
            i = rng.integers(0, len(pts))
            j = rng.integers(0, len(pts))
            if i == j:
                continue
            xi, fi = pts[i]; xj, fj = pts[j]
            t = rng.uniform(0.15, 0.85)
            cand = np.clip((1.0-t)*xi + t*xj, lo, hi)
            key = tuple(np.round(cand, 6))
            if key in used:
                continue
            used.add(key)
            f = eval_real(cand)
            add_point(cand, f)

        # --- 5) Remaining Budget ---
        while self.eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            try:
                b, bf = local_hill(x, steps=10)
            except Exception:
                break
            add_point(b, bf)

        return None
# EVOLVE-BLOCK-END
