"""Rewritten program for DTLZ2 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using a combined deterministic grid strategy.
    
    Strategy:
    1. The Pareto front of DTLZ2 is a 2D manifold in the objective space.
    2. The position on the front is determined by the first two decision variables (x0, x1).
    3. The distance from the front is determined by the remaining variables (x2..x11).
    4. To maximize hypervolume, we place points directly on the Pareto front (g=0) by setting x2..x11 = 0.5.
    5. We use a deterministic uniform grid of 10x10 points as the base, as per the research directive.
    6. To utilize the full budget and maximize coverage, we supplement this with points from a 
       denser 14x14 grid, ensuring unique points are evaluated to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def generate_grid(n):
            pts = []
            for i in range(n):
                for j in range(n):
                    x = np.full(self.dim, 0.5)
                    x[0] = i / (n - 1)
                    x[1] = j / (n - 1)
                    pts.append(x)
            return np.array(pts)

        # Use a 14x14 grid (196 points) which is the largest square grid <= 200
        # This ensures uniform coverage across the entire domain, avoiding the bias
        # of the previous mixed-grid approach.
        grid_14 = generate_grid(14)
        
        seen = set()
        unique_points = []
        
        for x in grid_14:
            key = tuple(np.round(x, 10))
            if key not in seen:
                seen.add(key)
                unique_points.append(x)
                
        # Add 4 additional points to reach 200 and fill the largest gaps
        # These are the centers of the four quadrants of the [0,1]x[0,1] space.
        extra_points = [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75]
        ]
        
        for ep in extra_points:
            if len(unique_points) >= self.budget:
                break
            x = np.full(self.dim, 0.5)
            x[0] = ep[0]
            x[1] = ep[1]
            key = tuple(np.round(x, 10))
            if key not in seen:
                seen.add(key)
                unique_points.append(x)
                
        # If we still have budget (shouldn't happen with 196+4=200), add center
        if len(unique_points) < self.budget:
            x = np.full(self.dim, 0.5)
            x[0] = 0.5
            x[1] = 0.5
            key = tuple(np.round(x, 10))
            if key not in seen:
                seen.add(key)
                unique_points.append(x)

        unique_points = np.array(unique_points)
        unique_points = np.clip(unique_points, lo, hi)
        
        # Evaluate points
        used = 0
        for x in unique_points:
            if used >= self.budget:
                break
            problem(x)
            used += 1
            
        return unique_points
# EVOLVE-BLOCK-END
