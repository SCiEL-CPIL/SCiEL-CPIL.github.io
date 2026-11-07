# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic space-filling lattice + local Nelder-Mead refinement for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _halton(dim, n, start=1):
        """Generate n points in [0,1]^dim using Halton sequence."""
        pts = np.zeros((n, dim))
        for i in range(n):
            idx = start + i
            for d in range(dim):
                # Find (d+1)-th prime
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
                base = primes[d % len(primes)]
                # Compute Halton value
                f = 1.0
                result = 0.0
                n_val = idx
                while n_val > 0:
                    f /= base
                    result += f * (n_val % base)
                    n_val //= base
                pts[i, d] = result
        return pts

    @staticmethod
    def _nelder_mead(func, x0, bounds, max_iter=50, step=0.1):
        """Simple Nelder-Mead for multi-objective: minimize sum of objectives."""
        lo, hi = bounds
        dim = len(x0)
        n = dim + 1
        simplex = np.zeros((n, dim))
        simplex[0] = x0.copy()
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += step * (hi[i - 1] - lo[i - 1])

        def obj_sum(x):
            f = func(x)
            return np.sum(f)

        fvals = np.array([obj_sum(x) for x in simplex])

        for _ in range(max_iter):
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            centroid = np.mean(simplex[:-1], axis=0)
            reflection = centroid + (centroid - simplex[-1])
            reflection = np.clip(reflection, lo, hi)
            f_ref = obj_sum(reflection)

            if f_ref < fvals[0]:
                expansion = centroid + 2 * (centroid - simplex[-1])
                expansion = np.clip(expansion, lo, hi)
                f_exp = obj_sum(expansion)
                if f_exp < f_ref:
                    simplex[-1] = expansion
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = reflection
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = reflection
                fvals[-1] = f_ref
            else:
                contraction = centroid + 0.5 * (simplex[-1] - centroid)
                contraction = np.clip(contraction, lo, hi)
                f_con = obj_sum(contraction)
                if f_con < fvals[-1]:
                    simplex[-1] = contraction
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = obj_sum(simplex[i])

        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Phase 1: Deterministic Halton lattice for initial coverage
        n_lattice = min(80, budget // 2)
        halton_pts = self._halton(dim, n_lattice)
        X = lo + (hi - lo) * halton_pts
        F = np.array([problem(x) for x in X])
        used = n_lattice

        # Phase 2: Local refinement on non-dominated points
        remaining = budget - used
        n_refine = min(10, remaining // 10)  # Refine top non-dominated points

        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        # Get non-dominated points
        non_dom_mask = np.array([not is_dominated(F[i], F) for i in range(len(F))])
        non_dom_indices = np.where(non_dom_mask)[0]

        # Sort non-dominated by sum of objectives (proxy for quality)
        if len(non_dom_indices) > 0:
            sums = np.sum(F[non_dom_indices], axis=1)
            sorted_non_dom = non_dom_indices[np.argsort(sums)]
            refine_indices = sorted_non_dom[:n_refine]
        else:
            # If all dominated, pick best by sum
            sums = np.sum(F, axis=1)
            refine_indices = np.argsort(sums)[:n_refine]

        # Perform local refinement
        for idx in refine_indices:
            if time.time() > deadline:
                break
            if used >= budget:
                break

            x0 = X[idx].copy()
            # Local Nelder-Mead: minimize sum of objectives
            x_best, f_sum_best = self._nelder_mead(problem, x0, (lo, hi), max_iter=30, step=0.15)
            f_best = problem(x_best)
            used += 1

            # Replace the original point if the refined point is better or non-dominated
            if not is_dominated(f_best, F):
                X[idx] = x_best
                F[idx] = f_best

        return X
# EVOLVE-BLOCK-END
