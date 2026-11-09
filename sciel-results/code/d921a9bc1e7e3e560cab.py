# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """Sobol sequence sampling + Nelder-Mead refinement of top hypervolume contributors."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        ref = np.array([3000.0, 0.06], dtype=float)

        # Track evaluations
        self._eval_count = 0
        self._problem = problem

        def eval_fn(x):
            self._eval_count += 1
            return self._problem(x)

        # Phase 1: Sobol sequence sampling
        n_sobol = int(min(budget * 0.6, 128))
        n_sobol = 1 << max(0, (n_sobol - 1).bit_length())
        if n_sobol > budget:
            n_sobol = 1 << max(0, (n_sobol // 2 - 1).bit_length())
        if n_sobol < 16:
            n_sobol = 16

        sampler = qmc.Sobol(d=dim, scramble=False)
        points_unit = sampler.random(n=n_sobol)
        points = lo + points_unit * span
        points = np.clip(points, lo, hi)

        F = np.array([eval_fn(x) for x in points], dtype=float)
        used = self._eval_count

        # Helper: compute hypervolume for 2D
        def hypervolume_2d(pts, ref_pt):
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            sorted_pts = pts[order]
            mask = (sorted_pts[:, 0] < ref_pt[0]) & (sorted_pts[:, 1] < ref_pt[1])
            sorted_pts = sorted_pts[mask]
            if len(sorted_pts) == 0:
                return 0.0
            hv = 0.0
            for i in range(len(sorted_pts)):
                f1 = sorted_pts[i, 0]
                f2 = sorted_pts[i, 1]
                if i + 1 < len(sorted_pts):
                    w1 = sorted_pts[i + 1, 0] - f1
                else:
                    w1 = ref_pt[0] - f1
                h2 = ref_pt[1] - f2
                if w1 > 0 and h2 > 0:
                    hv += w1 * h2
            return hv

        # Helper: compute contributions
        def compute_contributions(pts, ref_pt):
            n = len(pts)
            if n == 0:
                return np.array([])
            total_hv = hypervolume_2d(pts, ref_pt)
            contribs = np.zeros(n)
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                if np.any(mask):
                    hv_without = hypervolume_2d(pts[mask], ref_pt)
                else:
                    hv_without = 0.0
                contribs[i] = total_hv - hv_without
            return contribs

        # Phase 2: Nelder-Mead refinement on top contributors
        remaining = budget - used
        if remaining > 5:
            contribs = compute_contributions(F, ref)
            n_refine = min(10, len(F))
            
            if n_refine > 0:
                top_indices = np.argsort(-contribs)[:n_refine]
                
                for idx in top_indices:
                    if self._eval_count >= budget - 1:
                        break
                    
                    x0 = points[idx].copy()
                    
                    def scalar_obj(x):
                        x = np.clip(x, lo, hi)
                        f = np.array(eval_fn(x), dtype=float)
                        if f[0] >= ref[0] or f[1] >= ref[1]:
                            return 1e6 + f[0] + f[1]
                        return f[0] + f[1]
                    
                    # Limit iterations to stay within budget
                    max_iter = min(15, remaining - 5)
                    max_iter = max(1, max_iter)
                    
                    try:
                        result = minimize(
                            scalar_obj,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                        )
                        
                        if self._eval_count < budget:
                            x_best = np.clip(result.x, lo, hi)
                            f_best = np.array(eval_fn(x_best), dtype=float)
                            points = np.vstack([points, x_best])
                            F = np.vstack([F, f_best])
                    except Exception:
                        break
                    
                    remaining = budget - self._eval_count
                    if remaining <= 5:
                        break

        return points
# EVOLVE-BLOCK-END
