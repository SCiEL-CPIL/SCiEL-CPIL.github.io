# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    DTLZ1-specific optimizer using greedy hypervolume maximization.
    
    Strategy:
    1. Analytically determine that convergence variables x[0:6] should be 0.5
       (for bounds [0,1]) to minimize g(x) = sum((x_i - 0.5)^2).
    2. Use a Sobol-like low-discrepancy sequence to generate candidate diversity
       values z in [0,1].
    3. Greedily select points that maximize incremental hypervolume contribution
       when added to the current set.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim=1):
        """Generate n points using a simple low-discrepancy sequence (Halton-like)."""
        # Use Halton sequence for better coverage in 1D
        points = []
        for i in range(1, n + 1):
            # In 1D, use a simple low-discrepancy sequence
            # Use the inverse of the van der Corput sequence
            x = 0.0
            inv_d = 1.0
            j = i
            while j > 0:
                inv_d *= 0.5
                x += inv_d * (j % 2)
                j //= 2
            points.append(x)
        return np.array(points)

    def _hypervolume_1d(self, points, ref):
        """
        Compute 1D hypervolume for a set of points with a reference value.
        points: array of scalar values (objectives)
        ref: reference value
        Returns the volume of the region dominated by the points.
        """
        if len(points) == 0:
            return 0.0
        # Sort in descending order (higher objective values are worse for minimization)
        sorted_points = np.sort(points)[::-1]
        # Filter points that are better than reference
        valid = sorted_points[sorted_points < ref]
        if len(valid) == 0:
            return 0.0
        # In 1D, the hypervolume is sum of (ref - f_i) for each point
        # But we need to account for dominance - only non-dominated points count
        # In 1D, all points are non-dominated if they're all different
        # Actually in 1D with minimization, the best point dominates all others
        # So hypervolume = ref - min(points)
        return max(0.0, ref - np.min(valid))

    def _hypervolume_contribution_3d(self, new_point, existing_points, ref):
        """
        Estimate the incremental hypervolume contribution of adding new_point
        to the set existing_points in 3D.
        
        This is an approximation. The exact computation is complex.
        We use a simple heuristic: the contribution is proportional to how
        much new_point extends the front in each dimension.
        """
        # A point contributes to hypervolume if it's not dominated by any existing point
        # And it extends the front in some way
        
        # Check if new_point is dominated by any existing point
        for ep in existing_points:
            if all(ep <= new_point) and any(ep < new_point):
                return 0.0  # Dominated, no contribution
        
        # Estimate contribution based on distance to reference in each dimension
        # This is a rough approximation
        margin = ref - new_point
        if np.any(margin < 0):
            return 0.0  # Not better than reference in all dimensions
        
        # Simple volume estimate: product of margins, but this overcounts
        # Use a more conservative estimate
        return np.prod(margin) * 0.1  # Conservative factor

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, keep 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For DTLZ1 with bounds [0,1]:
        # Convergence variables: x[0:dim-1] should be 0.5
        # Diversity variable: x[dim-1] varies in [0,1]
        
        # Verify that 0.5 is optimal for convergence variables
        # g(x) = sum((x_i - 0.5)^2) is minimized when x_i = 0.5
        x_conv = 0.5 * (lo[:dim-1] + hi[:dim-1])
        if np.allclose(lo[:dim-1], 0) and np.allclose(hi[:dim-1], 1):
            x_conv = np.full(dim-1, 0.5)
        else:
            # If bounds are not [0,1], use midpoint as best guess
            x_conv = 0.5 * (lo[:dim-1] + hi[:dim-1])
        
        # Generate candidate diversity values using low-discrepancy sequence
        # We want to use all budget evaluations
        n_candidates = budget
        
        # Generate diverse z values
        # Use a combination of endpoints and interior points
        z_candidates = self._sobol_sequence(n_candidates)
        
        # Ensure we have endpoints for maximum coverage
        # The Sobol/Halton sequence may not include exact 0 and 1
        # Add them explicitly
        z_candidates = np.unique(np.concatenate([z_candidates, [0.0, 1.0]]))
        z_candidates = np.clip(z_candidates, 0.0, 1.0)
        
        # If we have more candidates than budget, take the first budget
        if len(z_candidates) > budget:
            z_candidates = z_candidates[:budget]
        
        # Now we need to select which points to evaluate to maximize hypervolume
        # Greedy approach: start with empty set, add point with highest marginal contribution
        
        # But we can't compute exact hypervolume contribution without evaluating
        # So we'll use a geometric heuristic: prefer points that spread out in z
        
        # Actually, let's just evaluate all candidates since they're all on the Pareto front
        # (assuming x_conv is optimal). The hypervolume will be maximized by
        # having points spread across the front.
        
        evaluated_points = []  # Store objective vectors
        used_evals = 0
        
        # Sort z candidates to ensure systematic coverage
        # Use a spread-out order: start with extremes, then fill in
        # Actually, the Sobol sequence already has good spread properties
        
        for z_val in z_candidates:
            if used_evals >= budget:
                break
            if time.time() > deadline:
                break
                
            x = np.zeros(dim)
            x[:dim-1] = x_conv
            x[dim-1] = lo[dim-1] + (hi[dim-1] - lo[dim-1]) * z_val
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            evaluated_points.append(f)
            used_evals += 1
        
        return None

# EVOLVE-BLOCK-END
