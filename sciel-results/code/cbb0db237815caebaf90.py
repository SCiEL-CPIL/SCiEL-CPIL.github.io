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
        
        # Generate points from a 10x10 grid
        def generate_grid(n):
            pts = []
            for i in range(n):
                for j in range(n):
                    x = np.full(self.dim, 0.5)
                    x[0] = i / (n - 1)
                    x[1] = j / (n - 1)
                    pts.append(x)
            return np.array(pts)

        grid_10 = generate_grid(10)
        grid_14 = generate_grid(14)
        
        # Combine and remove duplicates
        # We use a set of rounded tuples to identify unique points
        # Rounding to 10 decimals is sufficient for these rational coordinates
        seen = set()
        unique_points = []
        
        # Prioritize 10x10 grid as per directive
        for x in grid_10:
            key = tuple(np.round(x, 10))
            if key not in seen:
                seen.add(key)
                unique_points.append(x)
        
        # Add points from 14x14 grid that are not already seen
        for x in grid_14:
            key = tuple(np.round(x, 10))
            if key not in seen:
                seen.add(key)
                unique_points.append(x)
                
        # If we still have budget, add points from a 15x15 grid or other locations
        # 10x10 (100) + 14x14 (196) - 4 overlaps = 192 points.
        # We have 200 - 192 = 8 points left.
        if len(unique_points) < self.budget:
            grid_15 = generate_grid(15)
            for x in grid_15:
                if len(unique_points) >= self.budget:
                    break
                key = tuple(np.round(x, 10))
                if key not in seen:
                    seen.add(key)
                    unique_points.append(x)
                    
        # Clip all points to bounds
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
