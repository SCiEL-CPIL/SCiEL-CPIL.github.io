# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qnorm


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using a Bayesian Optimization framework
    with Gaussian Process surrogate and hypervolume-based Expected Improvement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Normalize bounds to [0, 1]
        def normalize(x):
            return (x - lo) / span

        def denormalize(z):
            return lo + z * span

        # Reference point for hypervolume
        ref_point = np.array([3.0, 5.0, 7.0])

        # State
        X_norm = []  # normalized decision variables
        F = []       # objective values
        used = 0

        # Initialize with Latin Hypercube Sampling
        n_init = 24
        rng = np.random.RandomState(42)

        for i in range(n_init):
            z = np.random.rand(self.dim)
            # LHS: use stratified sampling
            z_lhs = (np.arange(self.dim) + rng.rand(self.dim)) / self.dim
            z = z_lhs
            X_norm.append(z)
            x = denormalize(z)
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)
            used += 1

        # Track all evaluated points
        all_points = [f.copy() for f in F]

        # GP hyperparameters
        length_scales = np.ones(self.dim)
        noise = 1e-6
        signal_var = 1.0

        def gp_predict(X_train, y, X_test):
            """Gaussian Process prediction for one objective."""
            n_train = len(X_train)
            n_test = len(X_test)

            # RBF kernel
            def kernel(a, b):
                # Squared distance
                dist = np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** 2, axis=2)
                return signal_var * np.exp(-0.5 * dist) + noise * np.eye(n_train) if len(a) == n_train else signal_var * np.exp(-0.5 * dist)

            K = kernel(X_train, X_train)
            K = K + noise * np.eye(n_train)

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            except np.linalg.LinAlgError:
                # Add jitter
                K = K + 1e-6 * np.eye(n_train)
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                except np.linalg.LinAlgError:
                    return np.mean(y) + np.zeros(n_test), np.var(y) + np.ones(n_test) * 0.1

            K_s = kernel(X_train, X_test)
            mu = K_s.T @ alpha

            v = np.linalg.solve(L, K_s)
            K_ss = signal_var * np.ones((n_test, n_test))
            var = K_ss - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-10)

            return mu, var

        def compute_hypervolume(points, ref):
            """Compute hypervolume for 3D points using a simple grid-based approximation."""
            # Filter points dominated by reference
            valid = [p for p in points if np.all(p < ref)]
            if not valid:
                return 0.0

            valid = np.array(valid)

            # Simple grid-based HV computation
            # Use adaptive grid resolution based on number of points
            n_grid = 30

            mins = np.min(valid, axis=0)
            maxs = np.maximum(ref, np.max(valid, axis=0))

            # Create grid
            grids = [np.linspace(mins[i], maxs[i], n_grid) for i in range(3)]
            G0, G1, G2 = np.meshgrid(grids[0], grids[1], grids[2], indexing='ij')

            # For each grid cell, check if it's dominated by any point
            # A point (x, y, z) dominates cell if x <= cell_x, y <= cell_y, z <= cell_z
            # Actually for minimization: point p dominates region if p_i <= region_max_i
            # We want the volume of union of regions dominated by each point

            # Use inclusion-exclusion or simple occupancy
            dominated = np.zeros((n_grid, n_grid, n_grid), dtype=bool)

            for p in valid:
                # Find grid indices where point dominates
                # Point p dominates cell (i,j,k) if p[0] <= G0[i,j,k] and p[1] <= G1 and p[2] <= G2
                mask = (G0 >= p[0]) & (G1 >= p[1]) & (G2 >= p[2])
                dominated |= mask

            # Compute volume of dominated cells
            dx = grids[0][1] - grids[0][0] if n_grid > 1 else 1.0
            dy = grids[1][1] - grids[1][0] if n_grid > 1 else 1.0
            dz = grids[2][1] - grids[2][0] if n_grid > 1 else 1.0

            return np.sum(dominated) * dx * dy * dz

        def hv_contribution_3d(points, ref):
            """Compute individual hypervolume contributions for 3D points."""
            # For simplicity, return equal contributions for non-dominated points
            # This is a simplification; true HV contribution requires more work
            n = len(points)
            if n == 0:
                return np.array([])

            points = np.array(points)
            valid = np.all(points < ref, axis=1)
            if not np.any(valid):
                return np.zeros(n)

            # Simple approach: each non-dominated point gets equal share
            contrib = np.zeros(n)
            nd_count = 0
            for i in range(n):
                if not valid[i]:
                    continue
                is_nd = True
                for j in range(n):
                    if i == j or not valid[j]:
                        continue
                    if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd_count += 1

            if nd_count > 0:
                total_hv = compute_hypervolume(points[valid], ref)
                for i in range(n):
                    if valid[i]:
                        is_nd = True
                        for j in range(n):
                            if i == j or not valid[j]:
                                continue
                            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                                is_nd = False
                                break
                        if is_nd:
                            contrib[i] = total_hv / nd_count

            return contrib

        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break

            X_train = np.array(X_norm)
            F_arr = np.array(F)

            # Get GP predictions for each objective
            mus = np.zeros((len(X_train), self.n_obj))
            vars_ = np.zeros((len(X_train), self.n_obj))

            for m in range(self.n_obj):
                mu, var = gp_predict(X_train, F_arr[:, m], X_train)
                mus[:, m] = mu
                vars_[:, m] = var

            # Generate candidate points for acquisition
            n_candidates = 200
            candidates = np.random.rand(n_candidates, self.dim)

            # Predict at candidates
            cand_mus = np.zeros((n_candidates, self.n_obj))
            cand_vars = np.zeros((n_candidates, self.n_obj))

            for m in range(self.n_obj):
                mu, var = gp_predict(X_train, F_arr[:, m], candidates)
                cand_mus[:, m] = mu
                cand_vars[:, m] = var

            # Hypervolume-based Expected Improvement acquisition
            # For each candidate, estimate the HV improvement
            # Use a simplified EI: for each objective, compute EI and take minimum
            # (since we want improvement in all objectives)

            # Current best hypervolume
            current_hv = compute_hypervolume(F_arr, ref_point)

            # For each candidate, compute a score based on potential HV improvement
            # Use a simple approach: score = -max(mus) + exploration term
            # Better: use the minimum over objectives of (EI_m)

            acquisition = np.zeros(n_candidates)

            for m in range(self.n_obj):
                # Compute EI for objective m
                # Best value so far
                best_m = np.min(F_arr[:, m])

                # EI = (best - mu) * Phi((best - mu) / sigma) + sigma * phi((best - mu) / sigma)
                sigma = np.sqrt(cand_vars[:, m])
                z = (best_m - cand_mus[:, m]) / (sigma + 1e-10)
                ei = (best_m - cand_mus[:, m]) * 0.5 * (1 + qnorm(z)) + sigma * 0.5 * np.exp(-0.5 * z ** 2)
                ei = np.maximum(ei, 0)

                # Weight by current HV contribution potential
                acquisition += ei

            # Select best candidate
            best_idx = np.argmax(acquisition)
            new_z = candidates[best_idx]
            new_x = denormalize(new_z)
            new_x = np.clip(new_x, lo, hi)

            new_f = problem(new_x)
            X_norm.append(new_z)
            F.append(new_f)
            used += 1
            all_points.append(new_f.copy())

        return np.array(all_points)
# EVOLVE-BLOCK-END
