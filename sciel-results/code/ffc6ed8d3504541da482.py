"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE34 (5 variables, 3 objectives).
    Strategy:
    1. Initialize with a diverse set of points (Sobol-like + corners).
    2. Use a Kriging surrogate for each objective.
    3. Select next points via Expected Hypervolume Improvement (EHVI).
    4. Local refinement around promising points.
    5. Deadline guard to ensure we finish within the time limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([1705.0, 12.0, 0.3])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # State
        X = []  # List of evaluated points
        F = []  # List of objective vectors

        def evaluate(x):
            nonlocal X, F
            x = np.asarray(x)
            f = np.asarray(problem(x))
            X.append(x.copy())
            F.append(f.copy())
            return f

        def hypervolume(points, ref):
            """Compute exact hypervolume for small sets of points in 3D."""
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than ref in all objectives
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            # Compute hypervolume using inclusion-exclusion for 3D
            # For simplicity, use a straightforward O(n^3) approach for small n
            n = len(valid)
            if n == 1:
                return np.prod(ref - valid[0])
            # Sort by first objective
            valid = sorted(valid, key=lambda p: p[0])
            hv = 0.0
            # Use a simple recursive approach for 3D
            def hv_3d(pts, ref):
                if len(pts) == 0:
                    return 0.0
                if len(pts) == 1:
                    return np.prod(ref - pts[0])
                # Sort by first coordinate
                pts = sorted(pts, key=lambda p: p[0])
                # Partition into points that are dominated by the first point in obj 1
                # and those that aren't, then compute contributions
                # This is a simplified approach; for better performance, use a proper algorithm
                total = 0.0
                for i in range(len(pts)):
                    # Contribution of point i as the "corner"
                    # Points that are better than i in all other objectives
                    better = [j for j in range(len(pts)) if j != i and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i])]
                    if not better:
                        # This point is on the Pareto front in the remaining dimensions
                        width = (ref[0] - pts[i][0]) if i == 0 else (pts[i-1][0] - pts[i][0])
                        if width <= 0:
                            continue
                        # Compute hypervolume in remaining 2 dimensions
                        sub_pts = [p[1:] for p in pts if p[0] > pts[i][0] or p is pts[i]]
                        # Simpler: just compute the box contribution
                        box = (ref[0] - pts[i][0]) * (ref[1] - pts[i][1]) * (ref[2] - pts[i][2])
                        # Subtract overlaps - this is getting complex
                        # Let's use a different approach
                        pass
                # Fallback: use a simple approximation
                return 0.0
            
            # Use a simpler but correct approach for 3D hypervolume
            # Sort by first objective
            valid = sorted(valid, key=lambda p: p[0])
            hv = 0.0
            for i in range(len(valid)):
                # The contribution of point i is the volume of the box it uniquely covers
                # in the first dimension, times the hypervolume in the remaining dimensions
                # of points that are not dominated by earlier points in the first dimension
                x1_min = valid[i][0]
                x1_max = valid[i-1][0] if i > 0 else ref[0]
                if x1_max <= x1_min:
                    continue
                # Collect points with obj0 > x1_min (i.e., points i and later)
                sub_points = [p[1:] for p in valid[i:]]
                sub_ref = ref[1:]
                # Compute 2D hypervolume
                hv_2d = self._hypervolume_2d(sub_points, sub_ref)
                hv += (x1_max - x1_min) * hv_2d
            return hv

        def _hypervolume_2d(points, ref):
            """Compute 2D hypervolume."""
            if len(points) == 0:
                return 0.0
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            valid = sorted(valid, key=lambda p: p[0])
            hv = 0.0
            for i in range(len(valid)):
                x1_min = valid[i][0]
                x1_max = valid[i-1][0] if i > 0 else ref[0]
                if x1_max <= x1_min:
                    continue
                # The y-extent is the minimum y among points with x >= x1_min
                y_min = min(p[1] for p in valid[i:])
                if y_min < ref[1]:
                    hv += (x1_max - x1_min) * (ref[1] - y_min)
            return hv

        # Non-dominated set update
        def is_dominated(f, F_list):
            for g in F_list:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False

        def update_pareto_front(X_list, F_list):
            """Return indices of non-dominated points."""
            n = len(F_list)
            nd = [True] * n
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        nd[i] = False
                        break
            return [i for i in range(n) if nd[i]]

        # Kriging surrogate
        def kriging_predict(X_train, y_train, X_pred):
            """Simple Gaussian process prediction."""
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            X_pred = np.asarray(X_pred)
            if len(X_train) == 0:
                return np.zeros(X_pred.shape[0]), np.ones(X_pred.shape[0])
            # Compute distances
            d = np.linalg.norm(X_train[:, None, :] - X_pred[None, :, :], axis=2)
            # Use a simple exponential kernel
            sigma = 0.5
            K = np.exp(-d ** 2 / (2 * sigma ** 2))
            # Mean
            mu = y_train.mean()
            y_centered = y_train - mu
            # Solve K * alpha = y_centered
            try:
                alpha = np.linalg.solve(K + 1e-6 * np.eye(len(X_train)), y_centered)
            except:
                alpha = np.zeros(len(X_train))
            mean_pred = mu + K @ alpha
            var_pred = 1.0 - K @ K.T.diagonal()
            var_pred = np.maximum(var_pred, 1e-6)
            std_pred = np.sqrt(var_pred)
            return mean_pred, std_pred

        # EHVI acquisition function
        def ehvi(X_candidates, X_train, F_train, ref):
            """Compute Expected Hypervolume Improvement for candidate points."""
            n_cand = len(X_candidates)
            scores = np.zeros(n_cand)
            for i in range(n_cand):
                # Predict mean and std for each objective
                means = []
                stds = []
                for j in range(n_obj):
                    m, s = kriging_predict(X_train, F_train[:, j], X_candidates[i:i+1])
                    means.append(m[0])
                    stds.append(s[0])
                mean_f = np.array(means)
                std_f = np.array(stds)
                # Sample-based EHVI
                n_samples = 50
                improvements = []
                for _ in range(n_samples):
                    f_sample = mean_f + std_f * np.random.standard_normal(n_obj)
                    # Compute hypervolume with and without this point
                    base_hv = hypervolume(F_train, ref)
                    new_hv = hypervolume(F_train + [f_sample], ref)
                    improvements.append(max(0, new_hv - base_hv))
                scores[i] = np.mean(improvements)
            return scores

        # Initialization: diverse points
        n_init = min(20, budget)
        # Sobol-like initialization
        rng = np.random.default_rng(42)
        X_init = np.zeros((n_init, dim))
        for i in range(n_init):
            for d in range(dim):
                X_init[i, d] = lo[d] + (hi[d] - lo[d]) * ((i + 0.5) / n_init)
        # Add some random points
        n_random = min(10, budget - n_init)
        if n_random > 0:
            X_rand = lo + (hi - lo) * rng.random((n_random, dim))
            X_init = np.vstack([X_init, X_rand])
            n_init = len(X_init)

        # Evaluate initial points
        for x in X_init:
            evaluate(x)

        # Main loop
        used = len(X)
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find Pareto front
            pareto_idx = update_pareto_front(X, F)
            if len(pareto_idx) == 0:
                pareto_idx = [0]

            # Generate candidate points
            n_cand = 50
            candidates = []
            # Local perturbations around Pareto points
            for idx in pareto_idx[:10]:
                for _ in range(5):
                    step = 0.1 * (hi - lo)
                    cand = X[idx] + step * rng.standard_normal(dim)
                    cand = np.clip(cand, lo, hi)
                    candidates.append(cand)
            # Random candidates
            for _ in range(20):
                cand = lo + (hi - lo) * rng.random(dim)
                candidates.append(cand)

            candidates = np.array(candidates)

            # Compute EHVI scores
            if len(X) >= 5:
                scores = ehvi(candidates, np.array(X), np.array(F), self.ref_point)
                best_idx = np.argmax(scores)
            else:
                # Just pick random
                best_idx = rng.integers(len(candidates))

            x_new = candidates[best_idx]
            evaluate(x_new)
            used += 1

        return X
# EVOLVE-BLOCK-END
