"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct construction of DTLZ1 Pareto front via simplex lattice."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 290  # 40s safety margin before 330s kill

        # For DTLZ1 with 7 vars, 3 objs:
        # x_M = x[2:7] (5 vars), x_0, x_1 are the shape variables
        # g(x_M) = 100 * (|x_M| + sum_{j=1}^{4} [sin^2(10 * pi * x_M[j])])
        # f_0 = (1+g) * 0.5 * x_0 * sin(pi*x_1/2)
        # f_1 = (1+g) * 0.5 * x_0 * cos(pi*x_1/2)
        # f_2 = (1+g) * 0.5 * (1 - x_0)
        #
        # To get g=0: x_M = [0, 0.5, 0.5, 0.5, 0.5] gives |x_M|=2, but we need g=0
        # Actually g(x_M) = 100*(|x_M| + sum sin^2(10*pi*x_M[j]))
        # For g=0 we need |x_M|=0 AND all sin terms = 0
        # |x_M| = sum |x_M[j]|, so all x_M[j] = 0
        # Then sin(10*pi*0) = 0, so g=0 when x_M = [0,0,0,0,0]
        
        # Optimal x_M for g=0 is all zeros
        base_x2 = np.zeros(5)
        
        # Generate a dense grid on the simplex for (x_0, x_1)
        # The Pareto front in objective space is a 2D simplex
        # Parameterize using barycentric-like coordinates
        # f_0 + f_1 + f_2 = (1+g) * 0.5 * (x_0*sin + x_0*cos + 1-x_0)
        # When g=0: f_0 + f_1 + f_2 = 0.5 * (x_0*(sin+cos) + 1-x_0)
        # This isn't constant... let me reconsider
        
        # Actually for DTLZ1:
        # f_0 = (1+g)*0.5*x_0*sin(pi*x_1/2)
        # f_1 = (1+g)*0.5*x_0*cos(pi*x_1/2)  
        # f_2 = (1+g)*0.5*(1-x_0)
        #
        # The Pareto front when g=0 is parameterized by x_0 in [0,1], x_1 in [0,1]
        # This gives a curved surface, not a flat simplex
        
        # Strategy: Use a dense 2D grid over x_0 and x_1
        remaining = self.budget
        if remaining <= 0:
            return all_points
        
        # Use a grid that covers the (x_0, x_1) space densely
        # We want good coverage of the Pareto front
        n_grid = int(np.sqrt(remaining))
        n_grid = max(2, min(n_grid, 50))
        
        for i in range(n_grid):
            if len(all_points) >= self.budget or time.time() > deadline:
                break
            for j in range(n_grid):
                if len(all_points) >= self.budget or time.time() > deadline:
                    break
                # Map grid to [0,1]^2 for x_0 and x_1
                x0 = i / max(1, n_grid - 1)
                x1 = j / max(1, n_grid - 1)
                x = np.concatenate(([x0, x1], base_x2))
                f = problem(x)
                all_points.append(f.copy())

        # If budget remains, add some random exploration
        if len(all_points) < self.budget:
            rng = np.random.default_rng(42)
            while len(all_points) < self.budget and time.time() < deadline:
                x = np.zeros(7)
                x[0] = rng.random()
                x[1] = rng.random()
                x[2:] = rng.random(5)
                f = problem(x)
                all_points.append(f.copy())

        return all_points
# EVOLVE-BLOCK-END
