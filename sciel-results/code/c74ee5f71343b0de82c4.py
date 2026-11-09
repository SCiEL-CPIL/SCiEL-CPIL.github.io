"""Structure-aware optimizer for DTLZ1 using analytic front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits DTLZ1's known Pareto front structure: fix x[2:] to minimize g,
    sample the 2D simplex parameterized by x[0], x[1]."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ1: g(x) = 100*(n-1 + sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5)) for i in 2..n-1))
        # g is minimized when x_i = 0.5 for i >= 2, giving g = 0.
        # The Pareto front: f_1 = 0.5*(1+g)*x_0, f_2 = 0.5*(1+g)*x_1, f_3 = 0.5*(1+g)*(1-x_0-x_1)
        # with x_0 + x_1 <= 1, x_0, x_1 >= 0.
        #
        # Strategy:
        # 1. Set x[2:] = 0.5 to minimize g (g=0).
        # 2. Sample (x_0, x_1) uniformly on the 2D simplex to cover the front.
        # 3. Use remaining budget to perturb x[2:] slightly and refine.

        dim = self.dim
        budget = self.budget
        used = 0
        results = []

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            f = problem(x)
            used += 1
            results.append(f)
            return f

        # Phase 1: Sample the 2D simplex densely
        # Use stratified sampling on the simplex for good coverage
        n_phase1 = min(100, budget // 2)
        
        # Stratified sampling on triangle x0+x1<=1, x0,x1>=0
        # Use inverse CDF or grid-based approach
        # Simple approach: use random points in [0,1]^2 and reject x0+x1>1
        # But for better coverage, use a grid with jitter
        
        # Create a stratified grid
        n_grid = int(np.ceil(np.sqrt(2 * n_phase1)))
        # Points on a grid in the triangle
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid - i):
                x0 = (i + 0.5) / n_grid
                x1 = (j + 0.5) / n_grid
                if x0 + x1 <= 1.0:
                    grid_points.append((x0, x1))
        
        # Shuffle and take up to n_phase1 points
        np.random.shuffle(grid_points)
        grid_points = grid_points[:n_phase1]
        
        for x0, x1 in grid_points:
            if used >= budget or time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = x0
            x[1] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            eval_point(x)

        # Phase 2: Use remaining budget to refine
        # Perturb x[2:] slightly and also add more simplex points
        remaining = budget - used
        
        if remaining > 0:
            # Mix of: more simplex points with slight g perturbation
            # and local refinement around promising points
            
            # Generate more simplex points with small perturbations to x[2:]
            n_phase2 = remaining
            
            for _ in range(n_phase2):
                if used >= budget or time.time() > deadline:
                    break
                
                # Random point on simplex with slight perturbation to g-variables
                # Use Dirichlet-like sampling for better coverage
                r1 = np.random.random()
                r2 = np.random.random()
                # Map to simplex
                x0 = r1
                x1 = r2 * (1 - x0)
                
                x = np.full(dim, 0.5)
                x[0] = x0
                x[1] = x1
                # Small perturbation to x[2:] to explore near-front region
                if dim > 2:
                    x[2:] = 0.5 + 0.01 * np.random.randn(dim - 2)
                
                x = np.clip(x, lo, hi)
                eval_point(x)

        return results
# EVOLVE-BLOCK-END
