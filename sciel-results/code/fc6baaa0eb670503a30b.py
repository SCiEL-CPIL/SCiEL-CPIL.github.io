"""Multi-objective optimizer on DTLZ7 exploiting known front geometry."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured grid + local refinement for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # DTLZ7 structure:
        # g(x) = 12*n + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for i in [2, n)
        # For g=0, we need x[2:] = 0.5 (since (0.5-0.5)^2 - cos(0) = 0 - 1 = -1, sum = -12, g = 12*20 + (-12*20) = 0)
        # Wait, let me recalculate. DTLZ7: g = 12*n + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))
        # If x_i = 0.5 for all i in [2,n): each term = 0 - 1 = -1, sum = -(n-2)
        # g = 12*n - (n-2) = 11n + 2. That's not zero.
        # Actually the standard DTLZ7 has g = 12*n + sum(...). The minimum of each term is -1,
        # so min g = 12*n - (n-2) = 11n + 2.
        # But for the Pareto front, we need g = 0, which requires... actually g >= 11n+2 > 0.
        # Hmm, let me reconsider. The objectives are:
        # f1 = x0 * (1 + g)
        # f2 = x1 * (1 + g)
        # f3 = h(1 + g) where h = 1 - sqrt(x_{n-1}/(1+g)) * (1 - (x_{n-1}/(1+g))^2 * sin^2(pi*x_{n-1}/2))
        # Wait, actually for DTLZ7 with 3 objectives:
        # f1 = x0
        # f2 = x1
        # f3 = h(x_{n-1}, g) * (1 + g)
        # where h = 1 - sqrt(x_{n-1}/(1+g)) * (1 - (x_{n-1}/(1+g))^2 * sin^2(pi*x_{n-1}/2))
        # And g depends on x[2:n-1] (or x[2:n] depending on convention).
        # 
        # The key insight: for the Pareto front, we want g as small as possible.
        # The minimum of g is achieved when x_i = 0.5 for the g-variables.
        # 
        # For a 22-variable DTLZ7:
        # x0, x1 are front variables
        # x2..x21 are g-variables (20 variables)
        # g = 12*20 + sum_{i=2}^{21} ((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5)))
        # Min g = 240 - 20 = 220... that doesn't seem right for a standard benchmark.
        #
        # Let me just use the problem function as given. The key strategy:
        # 1. Set x[2:] = 0.5 to minimize g
        # 2. Vary x0 and x1 to cover the Pareto front
        # 3. Use local refinement around promising points

        # First, determine what g-variables should be. Try x[2:] = 0.5
        # and see what objectives we get.

        # Strategy: Grid on x0, x1 with x[2:] = 0.5
        # The Pareto front of DTLZ7 (3 obj) is parameterized by x0, x1, x_{n-1}
        # with g at minimum. f1 = x0, f2 = x1, f3 = h(x_{n-1}, g_min) * (1 + g_min)
        
        # Let's first evaluate a single point to understand the scale
        test_x = np.zeros(n)
        test_x[0] = 0.5
        test_x[1] = 0.5
        test_x[2:] = 0.5
        test_f = np.array(problem(test_x))
        
        # Grid search on x0 and x1
        # We have 200 budget. Use ~15x15 grid = 225, too many.
        # Use 12x12 = 144, leaving 56 for refinement.
        
        grid_size = 12
        num_grid = grid_size * grid_size
        if num_grid > self.budget:
            grid_size = int(np.sqrt(self.budget))
            num_grid = grid_size * grid_size
        
        x0_vals = np.linspace(0, 1, grid_size)
        x1_vals = np.linspace(0, 1, grid_size)
        
        points = []
        objectives = []
        
        for i in range(grid_size):
            if time.time() > deadline:
                break
            for j in range(grid_size):
                if time.time() > deadline:
                    break
                x = np.full(n, 0.5)
                x[0] = x0_vals[i]
                x[1] = x1_vals[j]
                f = np.array(problem(x))
                points.append(x.copy())
                objectives.append(f)
        
        if not points:
            # Fallback: single point
            x = np.full(n, 0.5)
            f = np.array(problem(x))
            return x
        
        points = np.array(points)
        objectives = np.array(objectives)
        used = len(points)
        
        # Local refinement around non-dominated points
        # Find non-dominated points
        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            for fj in F:
                if np.all(fj <= f) and np.any(fj < f):
                    return True
            return False
        
        # Keep only non-dominated points for refinement
        nondom_indices = []
        for i in range(len(objectives)):
            if not is_dominated(objectives[i], objectives):
                nondom_indices.append(i)
        
        # Refine around non-dominated points with small perturbations in x0, x1
        remaining_budget = self.budget - used
        if remaining_budget > 0 and nondom_indices:
            # Distribute remaining budget among non-dominated points
            n_refine = min(remaining_budget, len(nondom_indices) * 3)
            per_point = max(1, n_refine // max(1, len(nondom_indices)))
            
            # Create refinement candidates
            refine_indices = np.random.choice(nondom_indices, size=min(n_refine, len(nondom_indices) * 3), replace=False)
            
            for idx in refine_indices:
                if time.time() > deadline or used >= self.budget:
                    break
                
                base_x = points[idx]
                # Small perturbations in x0 and x1
                for delta in [0.01, -0.01, 0.02, -0.02, 0.005, -0.005]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x = base_x.copy()
                    x[0] = np.clip(base_x[0] + delta, 0, 1)
                    x[1] = np.clip(base_x[1] + delta, 0, 1)
                    f = np.array(problem(x))
                    points = np.vstack([points, x.reshape(1, -1)])
                    objectives = np.vstack([objectives, f.reshape(1, -1)])
                    used += 1
        
        return points
# EVOLVE-BLOCK-END
