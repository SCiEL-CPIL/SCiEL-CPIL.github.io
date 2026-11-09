"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Sobol sampling + Nelder-Mead refinement on top HV contributors."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([3000.0, 0.06])

        def evaluate(x):
            """Evaluate a single point, clipping to bounds."""
            x = np.clip(x, lo, hi)
            return np.array(problem(x), dtype=float)

        def hypervolume_2d(points):
            """Compute hypervolume in 2D with given reference point."""
            # Filter points strictly better than reference in both objectives
            mask = (points[:, 0] < ref[0]) & (points[:, 1] < ref[1])
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(pts)):
                x_val = pts[i, 0]
                y_val = pts[i, 1]
                if y_val < prev_y:
                    hv += (ref[0] - x_val) * (prev_y - y_val)
                    prev_y = y_val
                else:
                    # If y is not improving, we can break since sorted by x
                    # but actually we need to continue checking
                    pass
            return hv

        def hv_contribution(point, all_points):
            """Compute the hypervolume contribution of a single point."""
            # Hypervolume with point vs without
            hv_with = hypervolume_2d(np.vstack([point, all_points]))
            hv_without = hypervolume_2d(all_points)
            return hv_with - hv_without

        # Phase 1: Sobol sequence sampling
        n_sobol = min(128, budget - 20)  # leave some budget for local optimization
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Use a power of 2 for Sobol
        n_sobol = 1 << int(np.floor(np.log2(max(1, n_sobol))))
        if n_sobol > budget - 10:
            n_sobol = 64
        samples = sampler.random(n_sobol)
        X_sobol = lo + (hi - lo) * samples

        # Evaluate Sobol points
        F_list = []
        X_list = []
        used = 0
        for x in X_sobol:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            f = evaluate(x)
            X_list.append(x)
            F_list.append(f)
            used += 1

        X = np.array(X_list)
        F = np.array(F_list)

        # Phase 2: Select top points by hypervolume contribution and refine with Nelder-Mead
        # We'll use a scalarized objective: minimize weighted sum or use HV-based selection
        # For simplicity, select points that are non-dominated and spread out

        # Find non-dominated points
        def is_non_dominated(points):
            n = len(points)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                        dominated[i] = True
                        break
            return ~dominated

        if len(F) > 0:
            nd_mask = is_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]

            # If too many non-dominated, select top 10 by some criterion
            if len(nd_indices) > 10:
                # Select by distance from reference (closer is better)
                dists = np.linalg.norm(F[nd_indices] - ref, axis=1)
                top_indices = nd_indices[np.argsort(dists)[:10]]
            else:
                top_indices = nd_indices

            # Phase 3: Local refinement with Nelder-Mead
            remaining_budget = budget - used
            if remaining_budget > 0 and len(top_indices) > 0:
                n_refine = min(len(top_indices), 10)
                evals_per_refine = max(1, remaining_budget // n_refine)

                for idx in top_indices[:n_refine]:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break

                    x0 = X[idx]

                    # Define scalarized objective: minimize sum of objectives
                    # (this helps find points that are good in both)
                    def scalarized_obj(x):
                        x = np.clip(x, lo, hi)
                        f = np.array(problem(x), dtype=float)
                        return np.sum(f)

                    try:
                        result = minimize(
                            scalarized_obj,
                            x0,
                            method='Nelder-Mead',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': max(1, evals_per_refine),
                                     'xatol': 1e-4,
                                     'fatol': 1e-4}
                        )
                        if used < budget:
                            x_new = np.clip(result.x, lo, hi)
                            f_new = evaluate(x_new)
                            X = np.vstack([X, x_new])
                            F = np.vstack([F, f_new])
                            used += 1
                    except Exception:
                        pass

        # Return the best result found
        return X
# EVOLVE-BLOCK-END
