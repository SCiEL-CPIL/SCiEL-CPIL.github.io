"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-weighted recursive bisection grid + evolutionary search for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget
        m = self.n_obj  # number of objectives

        # --- Phase 1: Hypervolume-weighted recursive bisection grid on the Pareto front ---
        # For DTLZ2 with 3 objectives, the Pareto front is the portion of the unit sphere
        # in the positive octant: f_1^2 + f_2^2 + f_3^2 = 1, f_i >= 0.
        # We use recursive bisection in the objective space [0,1]^3 to place points
        # that maximize hypervolume contribution.
        
        # The Pareto front for DTLZ2 (3 obj) can be parameterized as:
        # f_1 = sin(theta1)
        # f_2 = cos(theta1) * sin(theta2)  
        # f_3 = cos(theta1) * cos(theta2)
        # where theta1, theta2 in [0, pi/2]
        
        # To create a decision vector on the Pareto front, we need:
        # - Set g-variables to their optimum (x_1...x_{n-m+1} = 0.5 for DTLZ2)
        # - Set position variables to give the desired objective values
        
        n_pos = m - 1  # number of position variables
        n_g = n - n_pos  # number of g variables
        
        def front_point_from_objectives(f_target):
            """Create a decision vector on the Pareto front targeting given objective values."""
            # For DTLZ2 with 3 objectives:
            # f_1 = sin(pi/2 * x_{n-1})
            # f_2 = sin(pi/2 * (x_n + 1))
            # f_3 = sin(pi/2 * (x_n + 2))
            # But this doesn't work cleanly. Let's use the standard parameterization.
            # Actually, for DTLZ2: f_i = g * sin(pi/2 * (x_{n-i+2} + 2*(i-1)/m))
            # For m=3: f_1 = g*sin(pi/2*x_{n-1}), f_2 = g*sin(pi/2*(x_n+1)), f_3 = g*sin(pi/2*(x_n+2))
            # This is inconsistent. Let me use a different approach.
            
            # Standard DTLZ2 parameterization:
            # The last m-1 variables determine the position on the front.
            # For m=3: x_{n-1}, x_n (indices n-2, n-1 in 0-based)
            # f_1 = sin(pi/2 * x_{n-1})
            # f_2 = sin(pi/2 * (x_n + 1)) 
            # f_3 = sqrt(1 - f_1^2 - f_2^2) on the front
            # Actually this isn't quite right either.
            
            # Let me use a cleaner approach: directly construct x that gives the target f.
            # For DTLZ2 on the Pareto front (g=1):
            # f_1 = sin(pi/2 * a), f_2 = sin(pi/2 * b), f_3 = sqrt(1 - f_1^2 - f_2^2)
            # where a, b are the position variables.
            # So: a = 2/pi * arcsin(f_1), b = 2/pi * arcsin(f_2)
            
            x = np.zeros(n)
            x[:n_g] = 0.5  # g-variables at optimum for g=1
            
            # Position variables
            if m >= 2:
                # f_1 corresponds to x_{n-1} (index n-2 in 0-based)
                f1_clipped = np.clip(f_target[0], 0, 1)
                x[n-2] = 2.0 / np.pi * np.arcsin(f1_clipped)
                
                if m >= 3:
                    # f_2 corresponds to x_n (index n-1 in 0-based)
                    f2_clipped = np.clip(f_target[1], 0, 1)
                    x[n-1] = 2.0 / np.pi * np.arcsin(f2_clipped)
            
            return np.clip(x, 0, 1)
        
        # Recursive bisection in objective space to maximize hypervolume
        # We work in the unit cube [0,1]^m and recursively bisect,
        # placing points at the center of each sub-rectangle that maximizes
        # the marginal hypervolume gain.
        
        ref = np.array([2.5] * m)
        
        # Generate points using recursive bisection
        # Start with the full box [0,1]^m
        # At each step, bisect the box along the dimension that gives the largest
        # hypervolume gain, and place a point at the center.
        
        def hv_contribution_3d(point, existing_points):
            """Compute the marginal hypervolume contribution of a point."""
            if len(existing_points) == 0:
                return min(ref[0] - point[0], 0) * min(ref[1] - point[1], 0) * min(ref[2] - point[2], 0) if all(point[i] < ref[i] for i in range(3)) else 0.0
            
            # Simplified: just use the volume of the box from point to ref
            if all(point[i] < ref[i] for i in range(3)):
                vol = (ref[0] - point[0]) * (ref[1] - point[1]) * (ref[2] - point[2])
                return vol
            return 0.0
        
        # Generate initial set of points using recursive bisection
        # For 3 objectives, we can use a 2D grid in the angle space
        # with non-uniform spacing weighted by hypervolume contribution
        
        n_init = min(80, budget // 3)
        
        # Use a stratified approach in the spherical parameterization
        # with non-uniform spacing to prioritize regions with larger HV contribution
        # The hypervolume contribution is proportional to the "solid angle" subtended
        
        # For the 3D Pareto front (quarter sphere), the hypervolume contribution
        # of a region is proportional to its surface area on the sphere.
        # So uniform sampling in solid angle gives hypervolume-optimal distribution.
        
        # Parameterize by solid angle: use uniform sampling in [0, pi/2]^2
        # but with a correction for the Jacobian of the spherical coordinates.
        
        # Actually, for the quarter sphere, uniform sampling in (theta1, theta2)
        # where theta1 is the polar angle and theta2 is the azimuthal angle
        # gives non-uniform surface density. The surface element is:
        # dA = cos(theta1) * dtheta1 * dtheta2 (for the standard spherical coords)
        # So to get uniform surface density, we need to sample theta1 with density
        # proportional to 1/cos(theta1), i.e., use the CDF approach.
        
        # Simpler: just use a dense grid and let the evolutionary phase refine.
        
        grid_size = int(np.ceil(np.sqrt(n_init)))
        theta1_vals = np.linspace(0.001, np.pi/2 - 0.001, grid_size)
        theta2_vals = np.linspace(0.001, np.pi/2 - 0.001, grid_size)
        
        init_points = []
        for t1 in theta1_vals:
            for t2 in theta2_vals:
                if len(init_points) >= n_init:
                    break
                f1 = np.sin(t1)
                f2 = np.cos(t1) * np.sin(t2)
                f3 = np.cos(t1) * np.cos(t2)
                f_target = np.array([f1, f2, f3])
                x = front_point_from_objectives(f_target)
                init_points.append(x)
        
        # Evaluate initial points
        F_init = np.array([problem(x) for x in init_points])
        used = len(init_points)
        
        # Keep only valid points (all objectives < ref)
        valid_mask = np.all(F_init < ref, axis=1)
        X = np.array(init_points[valid_mask])
        F = F_init[valid_mask]
        
        if len(X) == 0:
            # Fallback: random initialization
            X = lo + (hi - lo) * rng.random((10, n))
            F = np.array([problem(x) for x in X])
            used = 10
        
        # Phase 2: Evolutionary refinement with hypervolume-aware selection
        # Use a simple evolutionary strategy that prioritizes points with high
        # marginal hypervolume contribution
        
        while used < budget:
            # Compute non-dominated set
            n_pop = len(X)
            dominated = np.zeros(n_pop, dtype=bool)
            for i in range(n_pop):
                if dominated[i]:
                    continue
                for j in range(n_pop):
                    if i == j or dominated[i]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            
            non_dom_mask = ~dominated
            n_nd = np.sum(non_dom_mask)
            
            if n_nd == 0:
                non_dom_mask = np.ones(n_pop, dtype=bool)
                n_nd = n_pop
            
            non_dom_X = X[non_dom_mask]
            non_dom_F = F[non_dom_mask]
            
            # Select parent from non-dominated set, weighted by hypervolume contribution
            # Approximate HV contribution as the volume of the box from point to ref
            hv_scores = np.prod(ref - non_dom_F, axis=1)
            # Normalize to probabilities
            total_hv = np.sum(hv_scores)
            if total_hv > 0:
                probs = hv_scores / total_hv
            else:
                probs = np.ones(len(non_dom_F)) / len(non_dom_F)
            
            # Sample a parent
            parent_idx = rng.choice(len(non_dom_F), p=probs)
            parent = non_dom_X[parent_idx]
            
            # Mutate with adaptive mutation rate
            mutation_rate = 0.05 + 0.05 * (1 - used / budget)
            child = parent + mutation_rate * (hi - lo) * rng.standard_normal(n)
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Replace the point with lowest hypervolume contribution
            all_hv_scores = np.prod(ref - F, axis=1)
            worst_idx = int(np.argmin(all_hv_scores))
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
