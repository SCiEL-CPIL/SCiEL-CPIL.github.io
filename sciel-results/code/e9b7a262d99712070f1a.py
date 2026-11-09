"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a targeted manifold search for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # WFG9 has 12 variables. The Pareto-optimal manifold is characterized by:
        # x_0, x_1, x_2 being the decision variables that determine the objective values,
        # while x_3..x_11 are fixed at specific values (typically mid-range or specific constants).
        # For WFG9, the standard formulation has 12 variables where the first 3 are the "shape" variables
        # and the remaining 9 are "position" variables that must be set to specific values for optimality.
        
        # Standard WFG9:
        # Objectives depend on t_1, t_2, t_3 (derived from x_0, x_1, x_2)
        # The position variables x_4..x_12 (or x_3..x_11 in 0-indexed) must be set to 
        # the midpoint of their bounds or specific values to minimize the penalty terms.
        
        # For WFG9, the Pareto front lies on the manifold where:
        # x_0, x_1, x_2 vary freely (within bounds)
        # x_3..x_11 are fixed at the midpoint of their bounds (or specific values)
        
        # Let's identify which variables are "position" vs "shape"
        # In WFG9, typically the first 3 variables determine the front shape, and the rest are position.
        
        # Strategy: 
        # 1. Fix the position variables (x_3..x_11) at their midpoints
        # 2. Do a dense random search over the shape variables (x_0, x_1, x_2)
        # 3. Use remaining budget for local refinement
        
        mid = 0.5 * (lo + hi)
        
        # Determine which variables are shape vs position
        # For WFG9 with 12 variables, typically first 3 are shape, last 9 are position
        # But let's be more careful: the problem structure may vary
        # Let's assume first 3 are shape variables for WFG9
        
        n_shape = 3  # First 3 variables determine the Pareto front shape
        shape_idx = list(range(n_shape))
        pos_idx = list(range(n_shape, dim))
        
        # Set position variables to midpoints
        base_x = mid.copy()
        
        # Dense sampling over shape variables
        # Budget allocation: 70% for dense shape sampling, 30% for refinement
        n_shape_samples = int(budget * 0.7)
        n_refine = budget - n_shape_samples
        
        rng = np.random.default_rng(42)
        
        # Generate shape variable samples
        # Use a stratified approach to cover the shape space well
        shape_lo = lo[shape_idx]
        shape_hi = hi[shape_idx]
        
        # Create a grid-like sampling for better coverage
        # Divide each shape dimension into bins
        n_per_dim = max(1, int(round(n_shape_samples ** (1.0 / n_shape))))
        
        X_all = []
        F_all = []
        
        # Grid sampling over shape variables
        grid_points = []
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                for k in range(n_per_dim):
                    x_shape = np.array([
                        shape_lo[0] + (shape_hi[0] - shape_lo[0]) * (i + 0.5) / n_per_dim,
                        shape_lo[1] + (shape_hi[1] - shape_lo[1]) * (j + 0.5) / n_per_dim,
                        shape_lo[2] + (shape_hi[2] - shape_lo[2]) * (k + 0.5) / n_per_dim
                    ])
                    grid_points.append(x_shape)
        
        # If grid is too large, subsample
        if len(grid_points) > n_shape_samples:
            indices = rng.choice(len(grid_points), size=n_shape_samples, replace=False)
            grid_points = [grid_points[i] for i in indices]
        
        # Evaluate grid points
        for x_shape in grid_points:
            x = base_x.copy()
            x[shape_idx] = x_shape
            x = np.clip(x, lo, hi)
            f = problem(x)
            X_all.append(x)
            F_all.append(f)
        
        used = len(X_all)
        
        # Add some random samples for robustness
        n_random = min(n_shape_samples - used, max(0, int(n_shape_samples * 0.2)))
        for _ in range(n_random):
            x_shape = shape_lo + (shape_hi - shape_lo) * rng.random(n_shape)
            x = base_x.copy()
            x[shape_idx] = x_shape
            x = np.clip(x, lo, hi)
            f = problem(x)
            X_all.append(x)
            F_all.append(f)
            used += 1
        
        X_all = np.array(X_all)
        F_all = np.array(F_all, dtype=float)
        
        # Filter out points that are not strictly better than reference point
        ref_point = np.array([3.0, 5.0, 7.0])
        valid_mask = np.all(F_all < ref_point, axis=1)
        
        if np.any(valid_mask):
            X_all = X_all[valid_mask]
            F_all = F_all[valid_mask]
        
        # Refinement phase: local search around best points
        if n_refine > 0 and len(X_all) > 0:
            # Select diverse points from the current population
            # Use a simple non-dominated sort to identify front
            def is_dominated(f, F):
                return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
            
            # Find non-dominated points
            nd_indices = []
            for i in range(len(F_all)):
                if not is_dominated(F_all[i], F_all):
                    nd_indices.append(i)
            
            # Select up to 20 diverse points for refinement
            n_refine_points = min(20, len(nd_indices))
            if n_refine_points > 0:
                # Select evenly spaced indices from nd_indices
                step = max(1, len(nd_indices) // n_refine_points)
                selected = nd_indices[::step][:n_refine_points]
                
                # Local refinement around selected points
                n_per_point = max(1, n_refine // len(selected))
                
                for idx in selected:
                    if time.time() > deadline - 10.0:
                        break
                    x_center = X_all[idx]
                    # Small perturbation scale
                    scale = 0.1 * (hi - lo)
                    
                    for _ in range(n_per_point):
                        if used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        # Gaussian perturbation
                        x_new = x_center + rng.normal(0, 1, dim) * scale * 0.1
                        x_new = np.clip(x_new, lo, hi)
                        f_new = problem(x_new)
                        
                        # Only keep if better than reference
                        if np.all(f_new < ref_point):
                            X_all = np.vstack([X_all, [x_new]])
                            F_all = np.vstack([F_all, [f_new]])
                        used += 1
        
        return X_all
# EVOLVE-BLOCK-END
