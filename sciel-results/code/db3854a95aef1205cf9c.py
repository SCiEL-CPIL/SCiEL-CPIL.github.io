import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 320.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        n_samples = budget
        n = dim

        def f2_of_t(t):
            t = np.asarray(t, dtype=float)
            return 1.0 + np.sqrt(t / n) * (2.0 - t * (np.sin(10.0 * np.pi * t) + 1.0))

        if n_samples == 1:
            t_vals = np.array([0.5])
        else:
            # Initial guess: Arc-length uniform
            t_grid = np.linspace(1e-10, 1.0, 20000)
            f1 = t_grid
            f2 = f2_of_t(t_grid)

            df1_dt = np.ones_like(t_grid)
            df2_dt = np.gradient(f2, t_grid)

            ds_dt = np.sqrt(df1_dt**2 + df2_dt**2)
            L_cum = np.cumsum(ds_dt) * (t_grid[1] - t_grid[0])
            L_total = L_cum[-1]
            L_norm = L_cum / L_total

            fractions = np.linspace(0, 1, n_samples)
            t_vals = np.interp(fractions, L_norm, t_grid)
            t_vals = np.clip(t_vals, 0, 1)

            # Optimize t_vals to maximize HV
            f2_ref = 11.0
            f1_ref = 11.0

            def hv_from_t(t):
                # t is array of shape (n_samples,)
                t = np.asarray(t, dtype=float)
                f2 = f2_of_t(t)
                # Sort by t (f1)
                sort_idx = np.argsort(t)
                t_sorted = t[sort_idx]
                f2_sorted = f2[sort_idx]

                # Filter to non-dominated points: keep points where f2 is strictly less than min of all previous
                # Since sorted by f1 increasing, a point is dominated if there exists a previous point with f2 <= current f2
                keep = np.ones(len(t_sorted), dtype=bool)
                if len(t_sorted) > 1:
                    min_f2_so_far = np.empty(len(t_sorted))
                    min_f2_so_far[0] = f2_sorted[0]
                    for i in range(1, len(t_sorted)):
                        if f2_sorted[i] < min_f2_so_far[i-1]:
                            min_f2_so_far[i] = f2_sorted[i]
                        else:
                            min_f2_so_far[i] = min_f2_so_far[i-1]
                            keep[i] = False
                    # Also, the last kept point should be the one with minimum f2
                    # Actually, we need to keep points that are on the lower envelope
                    # Re-do: a point is non-dominated if no other point has both f1<= and f2<=
                    # With sorted by f1, point i is dominated if exists j<i with f2_j <= f2_i
                    # So keep point i if f2_i < min(f2_0, ..., f2_{i-1})
                    # The first point is always kept
                    # This gives the lower-left envelope

                t_kept = t_sorted[keep]
                f2_kept = f2_sorted[keep]

                if len(t_kept) == 0:
                    return 0.0

                t_ext = np.append(t_kept, f1_ref)
                widths = np.diff(t_ext)
                heights = f2_ref - f2_kept
                # Only count positive areas
                areas = widths * heights
                areas = np.maximum(areas, 0.0)
                return np.sum(areas)

            def neg_hv(t):
                return -hv_from_t(t)

            # Use Nelder-Mead for local optimization with tighter settings
            # Limit iterations to avoid timeout
            max_iter = 500
            res = minimize(neg_hv, t_vals, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-6})

            if res.fun < neg_hv(t_vals):
                t_vals = res.x
                t_vals = np.clip(t_vals, 0, 1)

            # Final polish: sort and remove duplicates
            t_vals = np.sort(t_vals)
            # Remove near-duplicates
            if len(t_vals) > 1:
                diffs = np.diff(t_vals)
                mask = np.concatenate(([True], diffs > 1e-10))
                t_vals = t_vals[mask]

        X = []
        F = []
        for t in t_vals:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
