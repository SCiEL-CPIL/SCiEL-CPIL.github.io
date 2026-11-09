"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE34 using Sobol initialization and
    hypervolume-guided local search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim, rng):
        """Generate a Sobol-like space-filling sequence."""
        # Use a simple stratified sampling approach as Sobol approximation
        # that guarantees good space-filling properties
        points = []
        # Use Halton sequence for better distribution
        def halton(index, base):
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i //= base
                f /= base
            return result

        for i in range(1, n + 1):
            point = []
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
            for b in bases:
                point.append(halton(i, b))
            points.append(point)
        return np.array(points)

    def _hypervolume_contribution(self, point, others, ref):
        """Compute the hypervolume contribution of a point given others."""
        # For 3 objectives, compute the volume of the region dominated by point
        # but not dominated by any other point
        # This is an approximation - we compute the box volume and subtract overlaps
        
        if len(others) == 0:
            # Full box from point to reference
            return max(0.0, ref[0] - point[0]) * max(0.0, ref[1] - point[1]) * max(0.0, ref[2] - point[2])
        
        # Simple approximation: compute the box volume
        vol = max(0.0, ref[0] - point[0]) * max(0.0, ref[1] - point[1]) * max(0.0, ref[2] - point[2])
        
        # Subtract overlap with other points (crude approximation)
        # A point dominates another if it's better in all objectives
        overlap = 0.0
        for other in others:
            if np.all(other <= point) and np.any(other < point):
                # other dominates point, so point contributes nothing
                return 0.0
        
        # For points that don't dominate each other, estimate overlap
        for other in others:
            if not np.all(other <= point) and not np.all(point <= other):
                # Partial overlap - approximate as fraction
                overlap_x = max(0.0, min(ref[0], other[0]) - point[0])
                overlap_y = max(0.0, min(ref[1], other[1]) - point[1])
                overlap_z = max(0.0, min(ref[2], other[2]) - point[2])
                overlap += overlap_x * overlap_y * overlap_z
        
        return max(0.0, vol - overlap)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Phase 1: Sobol/Halton initialization for first 128 evaluations
        n_init = min(128, self.budget)
        X_init = self._sobol_sequence(n_init, self.dim, rng)
        X = lo + (hi - lo) * X_init
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Filter to only points that are strictly better than reference in all objectives
        valid_mask = np.all(F < ref, axis=1)
        
        # Maintain a list of non-dominated points
        # For efficiency, we'll track the Pareto front
        def is_dominated(point, others):
            """Check if point is dominated by any point in others."""
            for other in others:
                if np.all(other <= point) and np.any(other < point):
                    return True
            return False
        
        # Build initial Pareto front
        pareto_front = []
        for i in range(len(F)):
            if valid_mask[i]:
                if not is_dominated(F[i], pareto_front):
                    # Remove points dominated by F[i]
                    pareto_front = [p for p in pareto_front if not is_dominated(p, [F[i]])]
                    pareto_front.append(F[i])
        
        # Phase 2: Local search on top points by hypervolume contribution
        n_local = self.budget - used
        
        # Compute hypervolume contributions for Pareto front points
        if len(pareto_front) > 0:
            contributions = []
            for i, pf in enumerate(pareto_front):
                others = [pareto_front[j] for j in range(len(pareto_front)) if j != i]
                contrib = self._hypervolume_contribution(pf, others, ref)
                contributions.append(contrib)
            
            # Sort by contribution descending, take top 10
            sorted_indices = np.argsort(contributions)[::-1]
            top_indices = sorted_indices[:min(10, len(pareto_front))]
        else:
            top_indices = []
        
        # Local search: perturb top points with small axis-aligned steps
        step_size = 0.05  # Small step for local refinement
        best_contribution = max(contributions) if len(pareto_front) > 0 else 0
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            if len(top_indices) > 0:
                # Pick a random top point
                idx = top_indices[rng.integers(len(top_indices))]
                parent_F = pareto_front[idx]
                parent_X = X[np.where(np.all(F == parent_F, axis=1))[0][0]]
            else:
                # If no valid points, pick a random point
                idx = rng.integers(len(X))
                parent_X = X[idx]
                parent_F = F[idx]
            
            # Generate child with small axis-aligned perturbation
            child = parent_X.copy()
            axis = rng.integers(self.dim)
            direction = rng.choice([-1, 1])
            child[axis] += direction * step_size * (hi[axis] - lo[axis])
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Check if child improves hypervolume
            if np.all(fc < ref):
                # Check if child is non-dominated
                if not is_dominated(fc, pareto_front):
                    # Remove dominated points from front
                    pareto_front = [p for p in pareto_front if not is_dominated(p, [fc])]
                    pareto_front.append(fc)
                    
                    # Update top_indices
                    if len(pareto_front) > 0:
                        contributions = []
                        for i, pf in enumerate(pareto_front):
                            others = [pareto_front[j] for j in range(len(pareto_front)) if j != i]
                            contrib = self._hypervolume_contribution(pf, others, ref)
                            contributions.append(contrib)
                        
                        sorted_indices = np.argsort(contributions)[::-1]
                        top_indices = sorted_indices[:min(10, len(pareto_front))]
            
            # Adapt step size based on improvement
            if np.all(fc < ref) and not is_dominated(fc, pareto_front):
                step_size = max(0.01, step_size * 0.8)  # Shrink on success
            else:
                step_size = min(0.1, step_size * 1.2)  # Grow on failure
        
        return X
# EVOLVE-BLOCK-END
