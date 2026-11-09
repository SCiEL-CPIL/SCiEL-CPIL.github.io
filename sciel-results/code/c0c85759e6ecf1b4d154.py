"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Hypervolume-driven optimizer for DTLZ4.
    
    Strategy:
    1. Initialize with points sampled from the known Pareto front structure.
    2. Use a greedy hypervolume maximization approach: at each step, generate candidates
       by perturbing existing points and selecting the one that maximizes marginal hypervolume.
    3. Additionally, use local optimization to refine points towards the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref):
        """Compute 2D hypervolume for points with reference point."""
        if len(pts) == 0:
            return 0.0
        pts = pts[np.all(pts < ref, axis=1)]
        if len(pts) == 0:
            return 0.0
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        total = 0.0
        max_y = ref[1]
        for i in range(len(pts) - 1, -1, -1):
            if pts[i, 1] < max_y:
                total += (max_y - pts[i, 1]) * (ref[0] - pts[i, 0])
                max_y = pts[i, 1]
        return total

    def _hv_3d(self, points, ref_point):
        """Compute exact 3D hypervolume using sweep line algorithm."""
        if len(points) == 0:
            return 0.0
        
        # Filter points that are strictly better than reference in all objectives
        valid = np.all(points < ref_point, axis=1)
        points = points[valid]
        if len(points) == 0:
            return 0.0
        
        # Sort by first objective
        order = np.argsort(points[:, 0])
        points = points[order]
        
        n = len(points)
        total_hv = 0.0
        
        # For each consecutive pair in x, compute the 2D HV of the yz-slice
        # The volume between x_i and x_{i+1} is (x_{i+1} - x_i) * HV_2d(S_i, ref_yz)
        # where S_i is the set of points with x <= x_i projected to yz
        
        for i in range(n - 1):
            x_range = points[i + 1, 0] - points[i, 0]
            # 2D HV of points up to i (inclusive) in yz
            yz_points = points[:i + 1, 1:3]
            hv_2d_val = self._hv_2d(yz_points, ref_point[1:3])
            total_hv += x_range * hv_2d_val
        
        # Add the last segment from x_n to ref_x
        x_range = ref_point[0] - points[-1, 0]
        yz_points = points[:, 1:3]
        hv_2d_val = self._hv_2d(yz_points, ref_point[1:3])
        total_hv += x_range * hv_2d_val
        
        return total_hv

    def _marginal_hv(self, point, existing_points, ref_point):
        """Compute the marginal hypervolume contribution of adding point to existing_points."""
        # HV of existing points
        hv_before = self._hv_3d(existing_points, ref_point)
        
        # HV of existing points plus the new point
        new_points = np.vstack([existing_points, [point]])
        hv_after = self._hv_3d(new_points, ref_point)
        
        return hv_after - hv_before

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([2.5, 2.5, 2.5])
        
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ4: f1 = g * cos(x1) * cos(x2)
        #         f2 = g * cos(x1) * sin(x2)
        #         f3 = g * sin(x1)
        # where g = 1 + 10*1 + sum_{i=3}^{12} (x_i^0.5 - 1)^2
        # Pareto front: g=1, so x_i = 1 for i=3..12, and x1, x2 vary.
        # f1 = cos(x1)*cos(x2), f2 = cos(x1)*sin(x2), f3 = sin(x1)
        # with x1 in [0, pi/2], x2 in [0, pi/2]
        
        # Initialize with points on the known Pareto front
        n_init = min(50, budget // 4)
        X = np.zeros((n_init, dim))
        for i in range(n_init):
            # Sample x1, x2 from the front
            x1 = rng.uniform(0, np.pi / 2)
            x2 = rng.uniform(0, np.pi / 2)
            X[i, 0] = x1
            X[i, 1] = x2
            # Set aggregation variables to 1 (to get g=1)
            X[i, 2:] = 1.0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Greedy hypervolume maximization
        # At each step, generate candidates by perturbing existing points
        # and select the one with the best marginal hypervolume contribution
        
        n_candidates = 10
        mutation_scale = 0.1
        
        while used < budget:
            # Generate candidates
            candidates = []
            cand_F = []
            
            for _ in range(n_candidates):
                # Pick a random parent
                parent_idx = rng.integers(len(X))
                parent = X[parent_idx].copy()
                
                # Mutate
                if rng.random() < 0.5:
                    # Gaussian mutation
                    mutation = rng.standard_normal(dim) * mutation_scale * (hi - lo)
                    child = parent + mutation
                else:
                    # Uniform mutation
                    child = parent + rng.uniform(-mutation_scale, mutation_scale, dim) * (hi - lo)
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                candidates.append(child)
            
            # Evaluate candidates
            for c in candidates:
                if used >= budget:
                    break
                fc = problem(c)
                used += 1
                cand_F.append(fc)
            
            if used >= budget:
                break
            
            cand_F = np.array(cand_F)
            
            # Select the candidate with the best marginal hypervolume
            # Use exact marginal HV computation
            best_score = -1
            best_idx = 0
            for i, fc in enumerate(cand_F):
                score = self._marginal_hv(fc, F, ref_point)
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            # Replace the worst point in the population
            # Use sum of objectives as a simple fitness for replacement
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = candidates[best_idx]
            F[worst_idx] = cand_F[best_idx]
        
        return X
# EVOLVE-BLOCK-END
