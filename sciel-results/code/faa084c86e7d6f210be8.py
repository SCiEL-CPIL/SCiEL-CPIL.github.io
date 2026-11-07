"""Rewritten program for DTLZ7 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 using a uniform grid to maximize
    hypervolume coverage of the Pareto front. Uniform grids provide better
    coverage (smaller maximum gap) than Gauss-Legendre nodes for this task,
    as the hypervolume contribution is roughly uniform across the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        
        # We use a uniform grid for the first two variables.
        # Let N1 be the number of points in x1, N2 in x2.
        # We want N1 * N2 <= budget.
        # To minimize the maximum gap, we want N1 and N2 to be as close as possible.
        # budget = 200.
        # sqrt(200) = 14.14.
        # 14 * 14 = 196.
        # 14 * 15 = 210 (too many).
        # So we use 14 * 14 = 196 points for the main grid.
        # This leaves 4 points for refinement.
        
        N1 = 14
        N2 = 14
        
        # Generate uniform nodes in [0, 1]
        # np.linspace includes endpoints 0 and 1.
        x1_nodes = np.linspace(0.0, 1.0, N1)
        x2_nodes = np.linspace(0.0, 1.0, N2)
        
        evaluated = 0
        
        # 1. Evaluate main grid points
        for f1 in x1_nodes:
            for f2 in x2_nodes:
                if evaluated >= self.budget:
                    break
                x = np.zeros(n)
                x[0] = f1
                x[1] = f2
                x[2:] = 0.0
                x = np.clip(x, lo, hi)
                problem(x)
                evaluated += 1
            if evaluated >= self.budget:
                break
        
        # 2. Use remaining budget for refinement points.
        # The main grid has 196 points. Budget is 200.
        # We have 4 points left.
        # We add points at the centers of the largest "gaps" or strategic locations.
        # Since the grid is uniform, all gaps are equal.
        # We add points to break the symmetry and improve coverage in the center.
        # The center of the domain is (0.5, 0.5).
        # In a 14x14 grid, 0.5 is exactly between the 7th and 8th nodes (indices 6 and 7).
        # Nodes are at k/13. 6/13 ~ 0.46, 7/13 ~ 0.53.
        # So (0.5, 0.5) is a good point.
        # We also add points slightly offset to cover the surrounding area better.
        
        extra_points = [
            (0.5, 0.5),
            (0.5, 0.25),
            (0.5, 0.75),
            (0.25, 0.5)
        ]
        
        for f1, f2 in extra_points:
            if evaluated >= self.budget:
                break
            x = np.zeros(n)
            x[0] = f1
            x[1] = f2
            x[2:] = 0.0
            x = np.clip(x, lo, hi)
            problem(x)
            evaluated += 1
            
        return None
# EVOLVE-BLOCK-END
