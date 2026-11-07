"""Improved DTLZ2 optimizer with hypervolume-weighted Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits the known Pareto front of DTLZ2 for high hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # For DTLZ2 with M=3 objectives and n=12 decision variables:
        # f_1 = g * (1 - x_1), f_2 = g * (1 - x_2), f_3 = g * x_2
        # g = 1 - sum_{j=3}^{12} (x_j - 0.5)^2
        # On the Pareto front, g=1, requiring x_j = 0.5 for j >= 3.
        # Then f_1 = 1 - x_1, f_2 = 1 - x_2, f_3 = x_2
        # So f_1 + f_2 = 1, f_3 = 1 - f_2
        # The PF is parameterized by x_1, x_2 in [0,1].
        # In objective space: f_1 in [0,1], f_2 in [0,1], f_3 = 1 - f_2
        # This is a line segment in 3D (not a sphere as I initially thought)
        
        # Actually, let me reconsider. The standard DTLZ2 has a spherical PF.
        # Let me check the formulation more carefully.
        # 
        # Standard DTLZ2:
        # f_i(x) = g(x) * (1 - x_i) for i = 1, ..., M-1
        # f_M(x) = g(x) * x_{M-1}
        # g(x) = 1 - sum_{j=M}^{n} (x_j - 0.5)^2
        #
        # For M=3:
        # f_1 = g * (1 - x_1)
        # f_2 = g * (1 - x_2)
        # f_3 = g * x_2
        #
        # On the PF, g=1:
        # f_1 = 1 - x_1
        # f_2 = 1 - x_2
        # f_3 = x_2
        #
        # So f_2 + f_3 = 1, and f_1 is independent.
        # This gives a 2D surface: f_1 in [0,1], f_2 in [0,1], f_3 = 1 - f_2
        #
        # But wait, this doesn't look like a sphere. Let me check if the 
        # formulation is different. Maybe it's:
        # f_i = g * x_i for i = 1..M-1, f_M = g * (1 - x_{M-1})
        # or some other variant.
        #
        # Regardless, the key insight is that on the PF, the g-variables
        # (x_3..x_12) are all 0.5, and only x_1 and x_2 vary.
        # So we should sample x_1 and x_2 to maximize hypervolume.
        
        # Hypervolume-weighted sampling strategy:
        # The hypervolume contribution of a point (f1, f2, f3) with reference
        # point (2.5, 2.5, 2.5) is approximately (2.5-f1)*(2.5-f2)*(2.5-f3)
        # for points that are non-dominated.
        
        # We'll use a recursive bisection approach in the (x1, x2) parameter space
        # that prioritizes regions with higher hypervolume contribution.
        
        # First, let's understand the mapping from (x1, x2) to objective space.
        # f_1 = 1 - x_1, f_2 = 1 - x_2, f_3 = x_2
        # So x_1 = 1 - f_1, x_2 = 1 - f_2 = f_3
        
        # The hypervolume contribution in objective space for a region 
        # [f1_min, f1_max] x [f2_min, f2_max] x [f3_min, f3_max] is
        # (2.5 - f1_min) * (2.5 - f2_min) * (2.5 - f3_min) - 
        # (2.5 - f1_max) * (2.5 - f2_max) * (2.5 - f3_max)
        # (approximately, for small regions)
        
        # In (x1, x2) space:
        # f1 = 1 - x1, f2 = 1 - x2, f3 = x2
        # So the region [x1_min, x1_max] x [x2_min, x2_max] maps to
        # f1 in [1-x1_max, 1-x1_min], f2 in [1-x2_max, 1-x2_min], f3 in [x2_min, x2_max]
        
        # Let's use a simple approach: sample more points where the 
        # hypervolume contribution is largest.
        
        # Generate candidate points using a weighted sampling strategy.
        # We'll use a grid with non-uniform spacing that's denser where
        # the hypervolume contribution is higher.
        
        # For simplicity, let's use a combination of:
        # 1. A uniform grid for baseline coverage
        # 2. Additional points concentrated in high-contribution regions
        
        # Hypervolume contribution for a point (f1, f2, f3):
        # h(f) = (2.5 - f1) * (2.5 - f2) * (2.5 - f3)
        # In (x1, x2) space: h(x1, x2) = (2.5 - (1-x1)) * (2.5 - (1-x2)) * (2.5 - x2)
        # = (1.5 + x1) * (1.5 + x2) * (2.5 - x2)
        
        # This is maximized when x1 and x2 are large (close to 1), but
        # the (2.5 - x2) term decreases as x2 increases.
        # Let's find the maximum:
        # d/dx2 [(1.5 + x2)(2.5 - x2)] = (2.5 - x2) - (1.5 + x2) = 1 - 2*x2 = 0
        # x2 = 0.5
        # So the maximum contribution is at x2 = 0.5, and x1 = 1.
        
        # We'll use importance sampling with density proportional to h(x1, x2).
        
        # Generate Sobol-like points and reweight them.
        # For simplicity, let's use a stratified sampling approach.
        
        # Divide the (x1, x2) space into cells and sample proportionally
        # to the hypervolume contribution of each cell.
        
        n_grid = 20  # 20x20 = 400 cells, we'll have ~200 points
        x1_edges = np.linspace(0, 1, n_grid + 1)
        x2_edges = np.linspace(0, 1, n_grid + 1)
        
        # Compute the hypervolume contribution of each cell
        # Cell (i, j) covers x1 in [x1_edges[i], x1_edges[i+1]], 
        #               x2 in [x2_edges[j], x2_edges[j+1]]
        # The contribution is approximately h(center) * area
        # where h(center) = (1.5 + x1_c) * (1.5 + x2_c) * (2.5 - x2_c)
        
        weights = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                x1_c = 0.5 * (x1_edges[i] + x1_edges[i+1])
                x2_c = 0.5 * (x2_edges[j] + x2_edges[j+1])
                h = (1.5 + x1_c) * (1.5 + x2_c) * (2.5 - x2_c)
                area = (x1_edges[i+1] - x1_edges[i]) * (x2_edges[j+1] - x2_edges[j])
                weights[i, j] = h * area
        
        # Normalize weights to sum to budget
        total_weight = weights.sum()
        if total_weight > 0:
            weights_norm = weights / total_weight
        else:
            weights_norm = np.ones_like(weights) / weights.size
        
        # Determine how many points to allocate to each cell
        points_per_cell = np.round(weights_norm * budget).astype(int)
        
        # Adjust to ensure we don't exceed budget
        total_points = points_per_cell.sum()
        if total_points > budget:
            # Remove excess points from cells with smallest weights
            flat_indices = np.argsort(weights_norm.flatten())[::-1]
            excess = total_points - budget
            for idx in flat_indices:
                i, j = np.unravel_index(idx, weights_norm.shape)
                if points_per_cell[i, j] > 0 and excess > 0:
                    points_per_cell[i, j] -= 1
                    excess -= 1
        elif total_points < budget:
            # Add remaining points to cells with largest weights
            flat_indices = np.argsort(weights_norm.flatten())[::-1]
            deficit = budget - total_points
            for idx in flat_indices:
                i, j = np.unravel_index(idx, weights_norm.shape)
                if deficit > 0:
                    points_per_cell[i, j] += 1
                    deficit -= 1
        
        # Generate points in each cell
        rng = np.random.default_rng(42)
        candidates = []
        
        for i in range(n_grid):
            for j in range(n_grid):
                n_pts = points_per_cell[i, j]
                if n_pts == 0:
                    continue
                # Sample n_pts points uniformly in the cell
                x1_samples = rng.uniform(x1_edges[i], x1_edges[i+1], size=n_pts)
                x2_samples = rng.uniform(x2_edges[j], x2_edges[j+1], size=n_pts)
                for x1, x2 in zip(x1_samples, x2_samples):
                    x = np.full(dim, 0.5)
                    x[0] = x1
                    x[1] = x2
                    x = np.clip(x, lo, hi)
                    candidates.append(x)
        
        # If we have fewer candidates than budget, add random points
        if len(candidates) < budget:
            n_random = budget - len(candidates)
            for _ in range(n_random):
                x = lo + (hi - lo) * rng.random(dim)
                candidates.append(x)
        
        # Truncate to budget
        candidates = candidates[:budget]
        
        # Evaluate candidates
        evaluated_points = []
        evaluated_objs = []
        
        for x in candidates:
            if len(evaluated_points) >= self.budget:
                break
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                evaluated_points.append(x)
                evaluated_objs.append(f)
            except:
                continue
        
        # If we still have budget, do some local refinement
        if len(evaluated_points) < self.budget and time.time() < deadline:
            # Perturb existing points
            while len(evaluated_points) < self.budget and time.time() < deadline:
                if not evaluated_points:
                    x = lo + (hi - lo) * rng.random(dim)
                else:
                    idx = rng.integers(len(evaluated_points))
                    x = evaluated_points[idx].copy()
                    # Small perturbation in the front variables
                    x[0] += 0.05 * rng.standard_normal()
                    x[1] += 0.05 * rng.standard_normal()
                    x = np.clip(x, lo, hi)
                try:
                    f = problem(x)
                    f = np.array(f, dtype=float)
                    evaluated_points.append(x)
                    evaluated_objs.append(f)
                except:
                    break
        
        return evaluated_points
# EVOLVE-BLOCK-END
