"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for RE34 (5 vars, 3 objectives).
    Phase 1: Sobol quasi-random sampling for uniform coverage (first 150 evals).
    Phase 2: Hypervolume-contribution-based local search (remaining 50 evals).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_3d(self, points, ref):
        """
        Compute exact 3D hypervolume for a set of points (all better than ref).
        points: (N, 3) array
        ref: (3,) array
        Returns: hypervolume scalar.
        """
        if len(points) == 0:
            return 0.0
        # Sort by first objective (ascending, i.e., better first)
        idx = np.argsort(points[:, 0])
        p = points[idx]
        r = ref
        
        n = len(p)
        hv = 0.0
        for i in range(n):
            # Width in dimension 0
            f0_next = p[i + 1, 0] if i < n - 1 else r[0]
            width = f0_next - p[i, 0]
            if width <= 0:
                continue
            # 2D hypervolume of points i..n-1 in (f1, f2) space, ref (r[1], r[2])
            # Only consider points that are better than ref in f1 and f2
            sub = p[i:, [1, 2]]
            sub = sub[(sub[:, 0] < r[1]) & (sub[:, 1] < r[2])]
            if len(sub) == 0:
                continue
            # 2D hypervolume: sort by f1
            sub_idx = np.argsort(sub[:, 0])
            sub_sorted = sub[sub_idx]
            hv_2d = 0.0
            for j in range(len(sub_sorted)):
                f1_next = sub_sorted[j + 1, 0] if j < len(sub_sorted) - 1 else r[1]
                w1 = f1_next - sub_sorted[j, 0]
                if w1 <= 0:
                    continue
                # Max extent in f2 for points j..end
                max_f2 = np.max(sub_sorted[j:, 1])
                h2 = r[2] - max_f2
                if h2 <= 0:
                    continue
                hv_2d += w1 * h2
            hv += width * hv_2d
        return hv

    def _marginal_hv_contributions(self, F, ref):
        """
        Compute marginal hypervolume contribution of each point in the set.
        Returns array of contributions (same length as F).
        """
        n = len(F)
        if n == 0:
            return np.array([])
        # Filter to only points better than ref in all objectives
        valid_mask = np.all(F < ref, axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        contribs = np.zeros(n)
        if len(valid_indices) == 0:
            return contribs
        
        F_valid = F[valid_indices]
        total_hv = self._hypervolume_3d(F_valid, ref)
        
        for idx in valid_indices:
            mask = np.ones(len(F_valid), dtype=bool)
            mask[list(valid_indices).index(idx)] = False
            hv_without = self._hypervolume_3d(F_valid[mask], ref)
            contribs[idx] = total_hv - hv_without
        return contribs

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for RE34
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Phase 1: Sobol quasi-random sampling
        n_sobol = min(150, budget)
        n_local = budget - n_sobol
        
        X = []
        F = []
        
        # Generate Sobol sequence
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sampler.random(n_sobol)
        sobol_points = lo + (hi - lo) * sobol_points
        
        for i in range(n_sobol):
            x = sobol_points[i]
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
        
        X = np.array(X)
        F = np.array(F)
        
        # Phase 2: Hypervolume-contribution-based local search with adaptive step
        if n_local > 0 and len(X) > 0:
            # Start with a moderate step size
            base_scale = 0.15 * (hi - lo)
            current_scale = base_scale.copy()
            
            for step in range(n_local):
                # Compute marginal contributions
                contribs = self._marginal_hv_contributions(F, ref)
                
                # Find the point with the lowest contribution among valid points
                valid_mask = np.all(F < ref, axis=1)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) == 0:
                    # No valid points, just random exploration
                    new_x = lo + (hi - lo) * rng.random(dim)
                    new_f = np.array(problem(new_x))
                    X[0] = new_x
                    F[0] = new_f
                    continue
                
                # Among valid points, find the one with lowest contribution
                valid_contribs = contribs[valid_indices]
                worst_local_idx = int(np.argmin(valid_contribs))
                worst_idx = valid_indices[worst_local_idx]
                
                # Adaptive step: decrease if we've been improving, increase if stuck
                # Use a simple strategy: vary the scale based on step number
                if step < n_local // 2:
                    current_scale = base_scale * (1.0 + 0.5 * np.cos(np.pi * step / (n_local / 2)))
                else:
                    current_scale = base_scale * 0.5
                
                # Generate a Gaussian perturbation
                perturbation = current_scale * rng.standard_normal(dim)
                new_x = X[worst_idx] + perturbation
                new_x = np.clip(new_x, lo, hi)
                
                new_f = np.array(problem(new_x))
                
                # Only replace if the new point is valid (better than ref in all objectives)
                # or if it has higher contribution than the current worst
                if np.all(new_f < ref):
                    X[worst_idx] = new_x
                    F[worst_idx] = new_f
                else:
                    # Try a smaller step
                    small_perturbation = (current_scale * 0.3) * rng.standard_normal(dim)
                    small_new_x = X[worst_idx] + small_perturbation
                    small_new_x = np.clip(small_new_x, lo, hi)
                    small_new_f = np.array(problem(small_new_x))
                    if np.all(small_new_f < ref):
                        X[worst_idx] = small_new_x
                        F[worst_idx] = small_new_f
        
        return X
# EVOLVE-BLOCK-END
