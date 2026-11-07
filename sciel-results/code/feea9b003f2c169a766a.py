"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer for RE34 combining decomposition-based local search
    with an evolutionary population loop for coverage and refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        width = hi - lo

        ref = np.array([1705.0, 12.0, 0.3])

        X_archive = []
        F_archive = []
        eval_count = 0

        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return [1e10] * self.n_obj
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            if not np.all(np.isfinite(f)):
                f = np.full(self.n_obj, 1e10)
            eval_count += 1
            X_archive.append(x.copy())
            F_archive.append(f.copy())
            return f

        def non_dominated_front(F_list):
            """Return indices of non-dominated points in F_list."""
            n = len(F_list)
            if n == 0:
                return []
            F = np.array(F_list)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            return [i for i in range(n) if not dominated[i]]

        def update_archive(x, f):
            """Add point to archive if it is not dominated by existing points."""
            f = np.asarray(f, dtype=float)
            if not np.all(f < ref):
                return False
            for f_old in F_archive:
                if np.all(f_old <= f) and np.any(f_old < f):
                    return False
            X_archive.append(x.copy())
            F_archive.append(f.copy())
            return True

        # 1. Initial diverse sampling using Latin Hypercube
        n_init = min(30, self.budget)
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            lhs_points[:, d] = (perm + np.random.rand(n_init)) / n_init

        X_init = lo + lhs_points * width
        for x in X_init:
            if eval_count >= self.budget:
                break
            safe_problem(x)

        if eval_count >= self.budget:
            return X_archive

        # 2. Decomposition-based local search phase
        # Generate weight vectors on a simplex grid
        n_grid = 5
        weights = []
        for w1 in np.linspace(0, 1, n_grid):
            for w2 in np.linspace(0, 1 - w1, max(1, int((1 - w1) * n_grid) + 1)):
                w3 = 1 - w1 - w2
                if w3 >= -1e-10:
                    w3 = max(0, w3)
                    w = np.array([w1, w2, w3])
                    if np.sum(w) > 0:
                        w = w / np.sum(w)
                        weights.append(w)

        # Add extreme and center points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
        weights.append(np.ones(self.n_obj) / self.n_obj)

        # Remove duplicates
        unique_weights = []
        for w in weights:
            is_dup = False
            for uw in unique_weights:
                if np.allclose(w, uw, atol=0.05):
                    is_dup = True
                    break
            if not is_dup:
                unique_weights.append(w)
        weights = unique_weights
        np.random.shuffle(weights)

        remaining_budget = self.budget - eval_count
        safe_budget = remaining_budget - 10
        if safe_budget < 10:
            while eval_count < self.budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.rand(self.dim)
                safe_problem(x)
            return X_archive

        n_weights = len(weights)
        budget_per_weight = max(3, safe_budget // n_weights)

        for w in weights:
            if eval_count >= self.budget - 5:
                break
            if time.time() > deadline:
                break

            w_eff = w / ref

            # Find best starting point for this direction
            if F_archive:
                F_arr = np.array(F_archive)
                F_scaled = F_arr / ref
                scores = F_scaled @ w
                best_idx = np.argmin(scores)
                x0 = X_archive[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0

            # Try a few starting points
            n_starts = min(2, max(1, budget_per_weight // 5))

            for s in range(n_starts):
                if eval_count >= self.budget - 5:
                    break
                if time.time() > deadline:
                    break

                if s == 0:
                    x_start = x0
                else:
                    x_start = x0 + 0.1 * width * (np.random.rand(self.dim) - 0.5)
                    x_start = np.clip(x_start, lo, hi)

                def scalar_obj(x, w_eff=w_eff):
                    f = safe_problem(x)
                    return np.dot(w_eff, f)

                try:
                    max_iter = max(1, budget_per_weight // (2 * n_starts))
                    res = minimize(scalar_obj, x_start, method='L-BFGS-B',
                                   bounds=bounds,
                                   options={'maxiter': max_iter, 'maxls': 10, 'ftol': 1e-12})
                except Exception:
                    pass

        if time.time() > deadline:
            return X_archive

        # 3. Evolutionary refinement loop
        # Use the non-dominated archive as parents
        step_scale = 0.15
        min_step_scale = 0.01
        stagnation_count = 0
        max_stagnation = 8

        while eval_count < self.budget - 3:
            if time.time() > deadline:
                break

            nd_idx = non_dominated_front(F_archive)
            if not nd_idx:
                nd_idx = list(range(len(F_archive)))

            max_parents = min(len(nd_idx), 10)
            if len(nd_idx) > max_parents:
                parent_indices = np.random.choice(nd_idx, size=max_parents, replace=False)
            else:
                parent_indices = nd_idx

            n_offspring = min(max_parents, self.budget - eval_count - 3)
            if n_offspring <= 0:
                break

            new_points = []

            for _ in range(n_offspring):
                if eval_count >= self.budget - 3:
                    break
                if time.time() > deadline:
                    break

                pi = parent_indices[np.random.randint(len(parent_indices))]
                parent_x = X_archive[pi]

                child_x = parent_x + np.random.randn(self.dim) * (step_scale * width)
                child_x = np.clip(child_x, lo, hi)

                f_child = safe_problem(child_x)
                new_points.append((child_x, f_child))

            # Check for improvement
            if new_points:
                improved = False
                for cx, cf in new_points:
                    if update_archive(cx, cf):
                        improved = True
                        break

                if improved:
                    stagnation_count = 0
                    step_scale = min(step_scale * 1.1, 0.3)
                else:
                    stagnation_count += 1
                    if stagnation_count >= max_stagnation:
                        step_scale = max(step_scale * 0.7, min_step_scale)
                        stagnation_count = 0

        # 4. Final polish: L-BFGS-B on the non-dominated front
        if time.time() < deadline and eval_count < self.budget - 2:
            nd_idx = non_dominated_front(F_archive)
            if nd_idx:
                # Select up to 5 non-dominated points to polish
                n_polish = min(5, len(nd_idx), (self.budget - eval_count - 2) // 3)
                if n_polish > 0:
                    polish_indices = np.random.choice(nd_idx, size=n_polish, replace=False)
                    for idx in polish_indices:
                        if eval_count >= self.budget - 2:
                            break
                        if time.time() > deadline:
                            break

                        x0 = X_archive[idx].copy()
                        f0 = F_archive[idx].copy()

                        # Use Tchebycheff-style aggregation with the point's own weight
                        w = f0 / np.sum(f0) if np.sum(f0) > 0 else np.ones(self.n_obj) / self.n_obj
                        w_eff = w / ref

                        def tcheb_obj(x, w_eff=w_eff):
                            f = safe_problem(x)
                            return np.max(w_eff * f)

                        try:
                            max_iter = max(1, (self.budget - eval_count - 2) // n_polish)
                            res = minimize(tcheb_obj, x0, method='L-BFGS-B',
                                           bounds=bounds,
                                           options={'maxiter': max_iter, 'maxls': 10, 'ftol': 1e-12})
                        except Exception:
                            pass

        # 5. Final random fill
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            safe_problem(x)

        return X_archive
# EVOLVE-BLOCK-END
