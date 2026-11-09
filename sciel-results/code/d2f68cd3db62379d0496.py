"""Rewritten optimizer: deterministic multi-scale coordinate descent with time guard."""
import time
import numpy as np


class Optimizer:
    """Deterministic multi-scale coordinate descent for Büche-Rastrigin (f4)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube initialization for GP surrogate
        # Generate deterministic LHS points
        def generate_lhs(n, dim, seed=42):
            rng = np.random.RandomState(seed)
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                points[:, d] = (perm + rng.uniform(size=n)) / n
            # Map to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        n_lhc = min(40, budget // 5)
        lhc_points = generate_lhs(n_lhc, dim)

        # Also add structured points: center, corners subset
        structured = [np.zeros(dim)]
        # Add 4 strategic corners
        for mask in [0, 1, 2, 3]:
            pt = np.array([[-5.0, 5.0][(mask >> d) & 1] for d in range(dim)])
            structured.append(pt)

        all_initial = np.vstack([lhc_points, structured])
        # Ensure we don't exceed budget
        all_initial = all_initial[:min(len(all_initial), budget - 10)]

        X_obs = []
        y_obs = []
        calls_used = 0

        best_x = None
        best_f = np.inf

        for sp in all_initial:
            if calls_used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            f_val = func(sp)
            calls_used += 1
            X_obs.append(sp.copy())
            y_obs.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = sp.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1
            X_obs.append(best_x.copy())
            y_obs.append(best_f)

        # Convert to arrays
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)

        # GP surrogate helper functions
        def gp_predict(X, X_train, y_train, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6):
            """Simple GP prediction using RBF kernel."""
            n_train = X_train.shape[0]
            n_pred = X.shape[0]
            
            # Compute kernel matrix K(X_train, X_train)
            K_train = np.zeros((n_train, n_train))
            for i in range(n_train):
                for j in range(n_train):
                    dist2 = np.sum((X_train[i] - X_train[j]) ** 2)
                    K_train[i, j] = sigma_f ** 2 * np.exp(-dist2 / (2 * length_scale ** 2))
            K_train += sigma_n * np.eye(n_train)
            
            # Compute K(X_pred, X_train)
            K_pred = np.zeros((n_pred, n_train))
            for i in range(n_pred):
                for j in range(n_train):
                    dist2 = np.sum((X[i] - X_train[j]) ** 2)
                    K_pred[i, j] = sigma_f ** 2 * np.exp(-dist2 / (2 * length_scale ** 2))
            
            # Solve for weights
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                mean = K_pred @ alpha
                v = np.linalg.solve(L, K_pred.T)
                var = sigma_f ** 2 - np.sum(v ** 2, axis=1)
                var = np.maximum(var, 0)
            except np.linalg.LinAlgError:
                # Fallback: use nearest neighbor interpolation
                mean = np.zeros(n_pred)
                var = np.ones(n_pred) * sigma_f ** 2
                for i in range(n_pred):
                    dists = np.sum((X_train - X[i]) ** 2, axis=1)
                    nearest = np.argmin(dists)
                    mean[i] = y_train[nearest]
            
            return mean, var

        # Phase 2: Iterative GP-guided optimization
        x = best_x.copy()
        fx = best_f

        # Adaptive step sizes for exploration
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

        for step in step_sizes:
            if calls_used >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Use GP to guide search
            # Generate candidate points around current best
            n_candidates = min(8, (budget - calls_used) // 2)
            if n_candidates < 2:
                break

            # Generate candidates: GP-guided + coordinate directions
            candidates = []

            # Coordinate directions
            for d in range(dim):
                for direction in [1.0, -1.0]:
                    x_new = x.copy()
                    x_new[d] += direction * step
                    x_new = np.clip(x_new, -5.0, 5.0)
                    candidates.append(x_new)

            # GP-guided points: sample from posterior mean
            if len(X_obs) >= 3 and calls_used < budget - 5:
                # Generate a small grid around x
                n_grid = 5
                grid_points = []
                for d in range(dim):
                    for frac in [0.5, 1.0, 1.5, 2.0, 2.5]:
                        xp = x.copy()
                        xm = x.copy()
                        xp[d] += frac * step
                        xm[d] -= frac * step
                        xp = np.clip(xp, -5.0, 5.0)
                        xm = np.clip(xm, -5.0, 5.0)
                        grid_points.append(xp)
                        grid_points.append(xm)
                
                # Limit grid size
                grid_points = grid_points[:min(len(grid_points), 20)]
                
                if len(grid_points) > 0:
                    # Predict using GP
                    try:
                        X_grid = np.array(grid_points)
                        # Use current observations
                        length_scale = max(step, 0.1)
                        sigma_f = max(np.std(y_obs) if len(y_obs) > 1 else 1.0, 0.1)
                        mean, var = gp_predict(X_grid, X_obs, y_obs, length_scale=length_scale, sigma_f=sigma_f)
                        
                        # Select top candidates by predicted mean (minimization)
                        top_k = min(4, len(mean))
                        top_indices = np.argsort(mean)[:top_k]
                        for idx in top_indices:
                            candidates.append(grid_points[idx])
                    except Exception:
                        pass

            # Deduplicate and limit candidates
            unique_candidates = []
            seen = set()
            for c in candidates:
                key = tuple(np.round(c, 6))
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(c)

            # Evaluate candidates
            improved = False
            for c in unique_candidates:
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                f_new = func(c)
                calls_used += 1
                X_obs.append(c.copy())
                y_obs.append(f_new)
                if f_new < fx:
                    fx = f_new
                    x = c.copy()
                    improved = True

            # If no improvement, shrink step and continue
            if not improved:
                continue

        # Phase 3: Final fine-tuning with very small steps
        if calls_used < budget:
            fine_steps = [0.00005, 0.00002, 0.00001]
            for step in fine_steps:
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                for d in range(dim):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new

        return x, fx
