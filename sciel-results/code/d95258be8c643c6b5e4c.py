"""Combined MBEA: GP surrogate with EHVI acquisition for RE37 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-Based Evolutionary Algorithm using GP surrogates and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # 15s safety margin before 330s limit
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for hypervolume (from task description)
        ref_point = np.array([1.1, 1.3, 1.1])

        rng = np.random.default_rng(0)

        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("deadline exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        # --- GP Surrogate ---
        def matern_kernel(X, Y, length_scale=0.5, variance=1.0, nu=2.5):
            sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
            sq_dists = np.maximum(sq_dists, 0)
            dist = np.sqrt(sq_dists)
            kernel = np.sqrt(3) * dist / length_scale
            return variance * (1 + kernel) * np.exp(-kernel)

        class GPSurrogate:
            def __init__(self, X, Y, length_scale=0.5):
                self.X = X
                self.Y = Y
                self.mu = np.mean(Y)
                self.sigma = np.std(Y) + 1e-10
                self.Y_scaled = (Y - self.mu) / self.sigma
                self.length_scale = length_scale
                self.variance = 1.0
                self.K = matern_kernel(X, X, length_scale, self.variance) + 1e-6 * np.eye(X.shape[0])
                try:
                    self.L = np.linalg.cholesky(self.K)
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_scaled))
                except np.linalg.LinAlgError:
                    self.K += 1e-4 * np.eye(X.shape[0])
                    self.L = np.linalg.cholesky(self.K)
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_scaled))

            def predict(self, X_new):
                K_s = matern_kernel(X_new, self.X, self.length_scale, self.variance)
                v = K_s @ self.L
                means = self.mu + (v @ self.alpha) * self.sigma
                v_diag = np.sum(v**2, axis=1)
                variances = np.maximum(self.variance - v_diag, 1e-10) * self.sigma**2
                stds = np.sqrt(variances)
                return means, stds

        # --- Hypervolume computation (2D/3D) ---
        def hypervolume_3d(points, ref):
            """Compute hypervolume for 3D points relative to reference point."""
            # Filter points dominated by reference
            valid = []
            for p in points:
                if all(p[i] < ref[i] for i in range(3)):
                    valid.append(p)
            if len(valid) == 0:
                return 0.0
            # Sort by first objective
            valid.sort(key=lambda x: x[0])
            # Remove dominated points
            pareto = []
            for p in valid:
                dominated = False
                for q in pareto:
                    if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2] and (q[0] < p[0] or q[1] < p[1] or q[2] < p[2]):
                        dominated = True
                        break
                if not dominated:
                    pareto.append(p)
            if len(pareto) == 0:
                return 0.0
            # Compute hypervolume using inclusion-exclusion for 3D
            # For simplicity, use a grid-based approximation or exact 3D HV
            # Exact 3D HV:
            vol = 0.0
            n = len(pareto)
            for i in range(n):
                x0 = max(pareto[i][0], min(pareto[i-1][0] if i > 0 else 0))
                x1 = pareto[i][0]
                # For each point i, compute the volume of the box it uniquely contributes
                # This is complex; use a simpler approach
            # Alternative: use the standard algorithm
            # Sort by x
            pareto.sort(key=lambda x: x[0])
            # For each point, compute the volume of the box from (x_i, y_i, z_i) to (x_{i+1}, min(y_j for j>i where y_j < y_i), min(z_j for j>i where z_j < z_i))
            # This is still complex. Let's use a Monte Carlo or grid approach for EHVI, and exact for final scoring
            # For now, return a simple estimate
            if n == 1:
                p = pareto[0]
                return max(0, ref[0] - p[0]) * max(0, ref[1] - p[1]) * max(0, ref[2] - p[2])
            # Use a simple grid-based approximation
            n_grid = 20
            xs = np.linspace(min(p[0] for p in pareto), ref[0], n_grid)
            ys = np.linspace(min(p[1] for p in pareto), ref[1], n_grid)
            zs = np.linspace(min(p[2] for p in pareto), ref[2], n_grid)
            dx = xs[1] - xs[0] if n_grid > 1 else 0
            dy = ys[1] - ys[0] if n_grid > 1 else 0
            dz = zs[1] - zs[0] if n_grid > 1 else 0
            vol = 0.0
            for i in range(n_grid - 1):
                for j in range(n_grid - 1):
                    for k in range(n_grid - 1):
                        cx = (xs[i] + xs[i+1]) / 2
                        cy = (ys[j] + ys[j+1]) / 2
                        cz = (zs[k] + zs[k+1]) / 2
                        if all(cx >= p[0] or cy >= p[1] or cz >= p[2] for p in pareto):
                            continue
                        if all(cx < ref[0] and cy < ref[1] and cz < ref[2]):
                            vol += dx * dy * dz
            return vol

        # --- EHVI Acquisition Function ---
        def compute_ehvi(X_candidates, gps, pareto_set, ref):
            """Compute EHVI for a set of candidates."""
            n_cand = X_candidates.shape[0]
            ehvi = np.zeros(n_cand)
            # Sample from GP posteriors
            n_samples = 50
            # Predict means and stds for all candidates
            means_list = []
            stds_list = []
            for g in gps:
                m, s = g.predict(X_candidates)
                means_list.append(m)
                stds_list.append(s)
            means = np.stack(means_list, axis=1)  # (n_cand, n_obj)
            stds = np.stack(stds_list, axis=1)    # (n_cand, n_obj)
            
            # Current hypervolume
            hv_current = hypervolume_3d([list(p) for p in pareto_set], ref)
            
            for i in range(n_cand):
                hv_improvements = []
                for s in range(n_samples):
                    # Sample objective values from GP posterior
                    f_sample = means[i] + rng.standard_normal(n_obj) * stds[i]
                    # Add to Pareto set and compute HV
                    new_pareto = list(pareto_set) + [f_sample]
                    hv_new = hypervolume_3d([list(p) for p in new_pareto], ref)
                    hv_improvements.append(max(0, hv_new - hv_current))
                ehvi[i] = np.mean(hv_improvements)
            return ehvi

        # --- Phase 1: Initial DOE ---
        # Use a space-filling design for initial exploration
        n_init = min(40, self.budget // 3)
        
        # Generate Halton-like sequence
        def halton(n, dim):
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                inv_base = 1.0 / base
                f = inv_base
                for i in range(n):
                    points[i, d] = f
                    f = inv_base * f
                    while f >= 1.0:
                        f = f / base - np.floor(f / base)
                        f = inv_base * f
            return points

        halton_pts = halton(n_init, dim)
        X_init = lo + (hi - lo) * halton_pts
        
        # Add some special points
        specials = [lo, hi, (lo + hi) / 2.0]
        for s in specials:
            X_init = np.vstack([X_init, s.reshape(1, -1)])
        
        X_init = X_init[:n_init]

        model_X = []
        model_Y = []
        pareto_set = []

        for x in X_init:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            try:
                f = eval_real(x)
                model_X.append(x.copy())
                model_Y.append(f.copy())
                # Update Pareto set
                is_pareto = True
                for p in pareto_set:
                    if all(p[i] <= f[i] for i in range(n_obj)) and any(p[i] < f[i] for i in range(n_obj)):
                        is_pareto = False
                        break
                if is_pareto:
                    # Remove any existing points dominated by f
                    new_pareto = []
                    for p in pareto_set:
                        if not all(f[i] <= p[i] for i in range(n_obj)) or not any(f[i] < p[i] for i in range(n_obj)):
                            new_pareto.append(p)
                    new_pareto.append(f)
                    pareto_set = new_pareto
            except Exception:
                break

        if len(model_X) == 0:
            return None

        Xtr = np.vstack(model_X)
        Ytr = np.vstack(model_Y)

        # --- Phase 2: MBEA Loop ---
        gps = []
        for k in range(n_obj):
            try:
                gps.append(GPSurrogate(Xtr, Ytr[:, k], length_scale=0.5))
            except Exception:
                gps.append(None)

        # Generate candidate grid for acquisition
        n_grid = 15
        grid = np.array(np.meshgrid(*([np.linspace(lo[i], hi[i], n_grid)] * dim)), dtype=float)
        grid = grid.reshape(dim, -1).T

        # Also add random candidates
        n_random = 100
        X_random = rng.uniform(lo, hi, size=(n_random, dim))
        X_candidates = np.vstack([grid, X_random])

        while self.eval_count < self.budget and time.time() < deadline:
            # Refit GPs
            gps = []
            for k in range(n_obj):
                try:
                    gps.append(GPSurrogate(Xtr, Ytr[:, k], length_scale=0.5))
                except Exception:
                    gps.append(None)

            if all(g is None for g in gps):
                # Fallback: random search
                x_new = rng.uniform(lo, hi, size=dim)
                f_new = eval_real(x_new)
                model_X.append(x_new.copy())
                model_Y.append(f_new.copy())
                Xtr = np.vstack(model_X)
                Ytr = np.vstack(model_Y)
                # Update Pareto
                is_pareto = True
                for p in pareto_set:
                    if all(p[i] <= f_new[i] for i in range(n_obj)) and any(p[i] < f_new[i] for i in range(n_obj)):
                        is_pareto = False
                        break
                if is_pareto:
                    new_pareto = []
                    for p in pareto_set:
                        if not all(f_new[i] <= p[i] for i in range(n_obj)) or not any(f_new[i] < p_new[i] for i in range(n_obj)):
                            new_pareto.append(p)
                    new_pareto.append(f_new)
                    pareto_set = new_pareto
                continue

            # Compute EHVI
            try:
                ehvi = compute_ehvi(X_candidates, gps, pareto_set, ref_point)
                idx = np.argmax(ehvi)
                x_new = X_candidates[idx].copy()
            except Exception:
                # Fallback: random
                x_new = rng.uniform(lo, hi, size=dim)

            # Evaluate
            f_new = eval_real(x_new)
            model_X.append(x_new.copy())
            model_Y.append(f_new.copy())
            Xtr = np.vstack(model_X)
            Ytr = np.vstack(model_Y)

            # Update Pareto set
            is_pareto = True
            for p in pareto_set:
                if all(p[i] <= f_new[i] for i in range(n_obj)) and any(p[i] < f_new[i] for i in range(n_obj)):
                    is_pareto = False
                    break
            if is_pareto:
                new_pareto = []
                for p in pareto_set:
                    if not all(f_new[i] <= p[i] for i in range(n_obj)) or not any(f_new[i] < p[i] for i in range(n_obj)):
                        new_pareto.append(p)
                new_pareto.append(f_new)
                pareto_set = new_pareto

            # Update candidate grid (add some random points)
            if self.eval_count % 10 == 0:
                X_new_rand = rng.uniform(lo, hi, size=(50, dim))
                X_candidates = np.vstack([X_candidates, X_new_rand])
                # Trim if too large
                if X_candidates.shape[0] > 500:
                    X_candidates = X_candidates[-500:]

        return None
# EVOLVE-BLOCK-END
