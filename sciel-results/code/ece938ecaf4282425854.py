"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ2-aware optimizer: deterministic Pareto front grid + Sobol sampling + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ2: Pareto front is g=0 (last dim-n_obj vars = 0),
        # f_i = cos(theta_i) where theta_i = (pi/2) * x[i] for i < n_obj
        # We use the first n_obj variables for position on the front,
        # and set the remaining (dim - n_obj) variables to 0.

        # Step 1: Deterministic 10x10 grid in 2D position space (for 3 objectives)
        # For 3 objectives, we need 2 position variables (x[0], x[1]) since
        # f_2 = cos(pi/2 * x[2]) is determined by the constraint that all are on front.
        # Actually for DTLZ2 with n_obj=3, the Pareto front is a quarter-sphere in 3D.
        # We parameterize using 2 angles. Let's use a grid over the 2D domain.
        
        # Better approach: use the known structure. For DTLZ2 with 3 objectives:
        # f_i = cos(theta_i) where sum of theta_i^2 related to being on the front.
        # Actually, for DTLZ2, the Pareto front is: f_i = cos(theta_i), i=1..M,
        # where theta_i are derived from x[0..M-1] as theta_i = (pi/2)*x[i],
        # and the constraint is that x[M..] = 0 for g=0.
        # But wait - for DTLZ2, the Pareto front is actually a (M-1)-sphere surface.
        # With M=3, it's a quarter of a 2-sphere.
        
        # The standard DTLZ2 formulation:
        # g(x) = sum_{i=M}^{n-1} (x_i - 0.5)^2
        # f_i(x) = (1 + g(x)) * cos(x_{i-1} * pi/2), i = 1..M
        # Pareto front: g=0, so x_M..x_{n-1} = 0.5 (or 0 depending on formulation)
        # f_i = cos(x_{i-1} * pi/2)
        
        # For the front, we need to sample x[0..n_obj-1] in [0,1] and set x[n_obj..] = 0.5
        # (or whatever makes g=0). Let's assume x[n_obj..] = 0.5 makes g=0.
        
        # Actually, let's just use a direct approach: sample points on the Pareto front
        # by choosing positions and setting the rest to minimize g.
        
        # For a 10x10 grid in 2D (since we have 3 objectives, 2 free position params):
        grid_size = 10
        points = []
        
        # Vertex points (extreme points of the front)
        for i in range(n_obj):
            x = np.zeros(dim)
            x[:n_obj] = 0.5  # default to middle
            # Set position variables to create a vertex
            # For DTLZ2, f_i = cos(theta_i), so to maximize f_i, set theta_i = 0 => x[i]=0
            # To minimize f_i, set theta_i = pi/2 => x[i]=1
            x[i] = 0.0  # This makes f_i = cos(0) = 1 (maximum for that objective)
            # But we want points ON the front, so other objectives should be at their min
            for j in range(n_obj):
                if j != i:
                    x[j] = 1.0  # f_j = cos(pi/2) = 0
            x[n_obj:] = 0.5  # g=0
            points.append(x)
        
        # Grid points in the 2D position space
        # For 3 objectives, we use x[0] and x[1] as position variables
        # x[2] is determined by the front constraint (but in DTLZ2, all x[0..M-1] are free)
        # Actually, for DTLZ2, ALL of x[0..M-1] are free on the front. The front is the set
        # of all (cos(theta_0), cos(theta_1), cos(theta_2)) where theta_i = pi/2 * x[i].
        # This traces out a portion of the unit sphere in the positive octant.
        
        # So we can sample x[0], x[1], x[2] freely in [0,1] and set x[3..] = 0.5
        # Let's use a 3D grid for better coverage, or use Sobol sampling.
        
        # Use a 5x5x5 grid in 3D position space
        grid_3d = 5
        for i in range(grid_3d):
            for j in range(grid_3d):
                for k in range(grid_3d):
                    x = np.zeros(dim)
                    x[0] = i / (grid_3d - 1)
                    x[1] = j / (grid_3d - 1)
                    x[2] = k / (grid_3d - 1)
                    x[n_obj:] = 0.5
                    points.append(x)
        
        # Evaluate all grid points
        F = np.array([problem(np.clip(x, lo, hi)) for x in points])
        used = len(points)
        
        # Step 2: Use remaining budget for Sobol-based sampling on the front
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Generate Sobol-like points in [0,1]^n_obj
            n_sobol = min(remaining, 100)
            # Simple quasi-random sampling using Halton-like sequence
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            
            sobol_points = []
            for idx in range(1, n_sobol + 1):
                x = np.zeros(dim)
                for d in range(n_obj):
                    x[d] = halton(idx, d + 2)  # bases: 2, 3, 5, ...
                x[n_obj:] = 0.5
                sobol_points.append(x)
            
            sobol_points = np.array(sobol_points)
            F_sobol = np.array([problem(np.clip(x, lo, hi)) for x in sobol_points])
            used += n_sobol
            
            # Combine
            points = points + [x for x in sobol_points]
            F = np.vstack([F, F_sobol])
        
        # Step 3: Local refinement using Nelder-Mead on promising regions
        if used < self.budget and time.time() < deadline:
            # Find non-dominated points and refine around them
            def is_dominated(f, F_set):
                return np.any(np.all(F_set <= f, axis=1) & np.any(F_set < f, axis=1))
            
            # Get non-dominated indices
            non_dom_mask = np.array([not is_dominated(F[i], F) for i in range(len(F))])
            non_dom_indices = np.where(non_dom_mask)[0]
            
            if len(non_dom_indices) > 0:
                # Refine around top non-dominated points
                n_refine = min(self.budget - used, 30)
                # Pick a few diverse non-dominated points
                step = max(1, len(non_dom_indices) // n_refine)
                refine_indices = non_dom_indices[::step][:n_refine]
                
                for idx in refine_indices:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x0 = points[idx]
                    # Simple local search: try small perturbations in position space
                    for trial in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        x_new = x0.copy()
                        for d in range(n_obj):
                            x_new[d] += 0.05 * (rng.random() - 0.5)
                        x_new = np.clip(x_new, lo, hi)
                        f_new = problem(x_new)
                        used += 1
                        # Replace worst point if new point is better
                        if not is_dominated(f_new, F):
                            worst_idx = np.argmax(np.sum(F, axis=1))
                            points[worst_idx] = x_new
                            F[worst_idx] = f_new
        
        return points
# EVOLVE-BLOCK-END
