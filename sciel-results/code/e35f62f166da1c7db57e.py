"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using MOEA/D-inspired decomposition with L-BFGS-B."""

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

        # Reference point for scaling
        ref = np.array([1705.0, 12.0, 0.3])

        X_archive = []
        F_archive = []
        eval_count = 0

        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return [1e10, 1e10, 1e10]
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            if not np.all(np.isfinite(f)):
                f = np.array([1e10, 1e10, 1e10])
            eval_count += 1
            X_archive.append(x.copy())
            F_archive.append(f.copy())
            return f

        # 1. Initial diverse sampling
        n_init = 30
        if n_init > self.budget:
            n_init = self.budget

        # Latin Hypercube Sampling for initial diversity
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            points = (perm + np.random.rand(n_init)) / n_init
            lhs_points[:, d] = points

        X_init = lo + lhs_points * width

        for x in X_init:
            if eval_count >= self.budget:
                break
            safe_problem(x)

        if eval_count >= self.budget:
            return X_archive

        # 2. Generate a grid of weight vectors for decomposition
        # For 3 objectives, use a systematic grid
        # n_obj = 3, so weights are (w1, w2, w3) with w1+w2+w3=1, wi >= 0
        # Use a resolution that gives good coverage
        n_grid = 5  # 5 points per axis -> C(5+3-1, 3-1) = C(7,2) = 21 weight vectors
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

        # Add extreme points explicitly
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)

        # Add center
        weights.append(np.ones(self.n_obj) / self.n_obj)

        # Remove duplicates (approximately)
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

        # Shuffle for diversity in processing order
        np.random.shuffle(weights)

        # 3. For each weight, run a local optimization from the best starting point
        remaining_budget = self.budget - eval_count
        n_weights = len(weights)

        # Reserve budget for safety
        safe_budget = remaining_budget - 5
        if safe_budget < 10:
            while eval_count < self.budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.rand(self.dim)
                safe_problem(x)
            return X_archive

        # Budget per weight vector
        budget_per_weight = max(3, safe_budget // n_weights)

        for w in weights:
            if eval_count >= self.budget - 2:
                break
            if time.time() > deadline:
                break

            # Scale weights by inverse reference point to balance objective magnitudes
            w_eff = w / ref

            # Find best starting point in current archive for this direction
            if F_archive:
                F_arr = np.array(F_archive)
                F_scaled = F_arr / ref
                scores = F_scaled @ w
                best_idx = np.argmin(scores)
                x0 = X_archive[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0

            # Also try a few random starting points for diversity
            n_starts = min(3, max(1, budget_per_weight // 5))

            for s in range(n_starts):
                if eval_count >= self.budget - 2:
                    break
                if time.time() > deadline:
                    break

                if s == 0:
                    x_start = x0
                else:
                    # Perturb the best point or use random
                    if np.random.rand() < 0.5:
                        x_start = x0 + 0.1 * width * (np.random.rand(self.dim) - 0.5)
                        x_start = np.clip(x_start, lo, hi)
                    else:
                        x_start = lo + (hi - lo) * np.random.rand(self.dim)

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

        # 4. Final random fill if budget remains
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            safe_problem(x)

        return X_archive
# EVOLVE-BLOCK-END
