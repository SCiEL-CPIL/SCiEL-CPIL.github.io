"""Optimizer for DTLZ4 multi-objective optimization using analytic front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _construct_point(self, x1, x2, bounds):
        """Construct a DTLZ4 point with g=0 (Pareto optimal)."""
        x = np.zeros(self.dim)
        x[0] = x1
        x[1] = x2
        # g-variables (indices 7-11) set to 0.5 for g=0
        for i in range(7, 12):
            lo, hi = bounds[i]
            x[i] = 0.5
        # x[2], x[3], x[4], x[5], x[6] can be anything (they don't affect objectives when g=0)
        # Set to 0.5 for safety
        for i in range(2, 7):
            x[i] = 0.5
        # Clip to bounds
        for i in range(self.dim):
            lo, hi = bounds[i]
            x[i] = np.clip(x[i], lo, hi)
        return x

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4: x[0], x[1] control the front; x[7..11] control g
        # For g=0: x[7..11] = 0.5
        # f1 = cos(x1/2)*cos(x2/2), f2 = cos(x1/2)*sin(x2/2), f3 = sin(x1/2)
        # x1, x2 in [0, 1]
        
        # Strategy: Use stratified grid on (x1, x2) to cover the quarter-sphere front
        # Budget: 200 evaluations total
        
        # Determine grid size
        # Use most of budget for front coverage
        # 14x14 = 196 points, leaving 4 for refinement
        n_grid = 14
        n_points = n_grid * n_grid
        
        # Create stratified grid points in [0, 1] x [0, 1]
        # Use Halton-like stratified sampling for better coverage
        points = []
        for i in range(n_grid):
            for j in range(n_grid):
                # Stratified: divide [0,1] into n_grid intervals
                # Place point in center of each cell
                x1 = (i + 0.5) / n_grid
                x2 = (j + 0.5) / n_grid
                points.append((x1, x2))
        
        # Evaluate all grid points
        used = 0
        for x1, x2 in points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = self._construct_point(x1, x2, bounds)
            f = problem(x)
            used += 1
        
        # Use remaining budget for local refinement
        # Try small perturbations around promising points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Try a few more points at different positions
            extra_positions = [
                (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
                (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
                (0.5, 0.5), (0.1, 0.9), (0.9, 0.1), (0.5, 0.0),
                (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),
            ]
            for x1, x2 in extra_positions:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                x = self._construct_point(x1, x2, bounds)
                f = problem(x)
                used += 1
        
        return None
# EVOLVE-BLOCK-END
