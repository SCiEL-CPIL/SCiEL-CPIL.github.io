# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic space-filling + Nelder-Mead refinement for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2
            for i in range(n):
                f = 1.0
                r = 0.0
                j = i + 1
                while j > 0:
                    f /= base
                    r += f * (j % base)
                    j //= base
                points[i, d] = r
        return points

    def _is_pareto(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(np.array(f, dtype=np.float64))
            return x, np.array(f, dtype=np.float64)

        # --- Phase 1: Halton space-filling ---
        n_fill = min(100, budget)
        halton_pts = self._halton(n_fill, dim)
        for i in range(n_fill):
            if used >= budget or time.time() > deadline:
                break
            x = lo + halton_pts[i] * (hi - lo)
            evaluate(x)

        if used >= budget or time.time() > deadline:
            return np.array(all_X) if all_X else None

        F = np.array(all_F)
        X = np.array(all_X)

        # --- Phase 2: Nelder-Mead refinement on top-k non-dominated points ---
        # Get non-dominated points
        pareto_mask = self._is_pareto(F)
        pareto_indices = np.where(pareto_mask)[0]

        if len(pareto_indices) == 0:
            return np.array(all_X)

        # Sort by sum of objectives to get top candidates
        sums = F[pareto_indices].sum(axis=1)
        sorted_pareto = pareto_indices[np.argsort(sums)]

        # Refine top-k points with Nelder-Mead
        n_refine = min(15, len(sorted_pareto))
        evals_left = budget - used

        # Estimate cost per NM run: dim*(dim+1) initial + ~20*dim iterations
        cost_per_nm = dim * (dim + 1) + 30 * dim
        n_refine = min(n_refine, max(1, evals_left // max(1, cost_per_nm)))

        for idx in sorted_pareto[:n_refine]:
            if used >= budget or time.time() > deadline - 5:
                break

            x0 = X[idx].copy()

            # Multi-objective NM: minimize sum of objectives
            def obj_sum(x):
                if used >= budget or time.time() > deadline - 2:
                    return 1e10
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                used += 1
                all_X.append(x_clipped.copy())
                all_F.append(np.array(f, dtype=np.float64))
                return np.sum(f)

            # Run Nelder-Mead
            try:
                result = minimize(
                    obj_sum, x0, method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4,
                             'adaptive': True}
                )
                if used < budget and time.time() < deadline - 2:
                    x_best = np.clip(result.x, lo, hi)
                    f_best = problem(x_best)
                    used += 1
                    all_X.append(x_best.copy())
                    all_F.append(np.array(f_best, dtype=np.float64))
            except Exception:
                pass

        # --- Phase 3: If budget remains, do more Halton around best regions ---
        if used < budget and time.time() < deadline - 5:
            # Get current best non-dominated points
            F_all = np.array(all_F)
            X_all = np.array(all_X)
            pareto_mask2 = self._is_pareto(F_all)
            pareto_idx2 = np.where(pareto_mask2)[0]
            if len(pareto_idx2) > 0:
                sums2 = F_all[pareto_idx2].sum(axis=1)
                top_idx2 = pareto_idx2[np.argsort(sums2)[:5]]
                for tidx in top_idx2:
                    if used >= budget or time.time() > deadline - 3:
                        break
                    center = X_all[tidx]
                    n_local = min(20, budget - used)
                    for _ in range(n_local):
                        if used >= budget or time.time() > deadline - 1:
                            break
                        # Random perturbation around center
                        scale = 0.1
                        x_new = center + np.random.randn(dim) * scale * (hi - lo)
                        x_new = np.clip(x_new, lo, hi)
                        evaluate(x_new)

        return np.array(all_X) if all_X else None

# EVOLVE-BLOCK-END
