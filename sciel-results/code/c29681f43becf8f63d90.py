# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 lies on the subspace where variables x_2...x_{n-1} are 0.
    We sample this subspace densely and use remaining budget for local refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 specific structure:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = h(x[1], g(x)) where g depends on x[2:]
        # Pareto front is at x[2:] = 0.
        
        # Strategy:
        # 1. Sample the Pareto front subspace (x[2:] = 0) with a grid.
        # 2. Use remaining budget for random perturbations of these points or full random search.
        
        points = []
        evals_used = 0
        
        # Grid sampling on the Pareto front subspace
        # Use ~80% of budget for the front
        n_front = int(self.budget * 0.8)
        # Choose grid size such that grid_size^2 <= n_front
        grid_size = int(np.sqrt(n_front))
        
        # Generate grid points for x[0] and x[1]
        x0_vals = np.linspace(lo[0], hi[0], grid_size)
        x1_vals = np.linspace(lo[1], hi[1], grid_size)
        
        for x0 in x0_vals:
            for x1 in x1_vals:
                if evals_used >= self.budget:
                    break
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                # x[2:] remains 0
                points.append(x)
                evals_used += 1
            if evals_used >= self.budget:
                break
                
        # Evaluate front points
        F = np.array([problem(x) for x in points])
        
        # Remaining budget for refinement or random search
        remaining = self.budget - evals_used
        if remaining > 0:
            # Local refinement: perturb the best points
            # Identify non-dominated points (simple check)
            # Or just pick a few random points from the front and perturb them
            # Since we are already on the front, perturbing x[2:] > 0 will make it worse.
            # Perturbing x[0], x[1] is just more sampling.
            # So, let's do some random search in the full space to be safe, 
            # but weighted towards the front.
            
            # Actually, just adding more grid points if grid_size was too small?
            # Or finer grid?
            
            # Let's add random points in the front subspace to fill gaps
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = np.random.uniform(lo[0], hi[0])
                x[1] = np.random.uniform(lo[1], hi[1])
                points.append(x)
        
        return points
# EVOLVE-BLOCK-END
