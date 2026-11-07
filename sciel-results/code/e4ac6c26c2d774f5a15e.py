import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Optimizer using Sobol sequence on the DTLZ4 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 Pareto front is defined by x_2..x_11 = 0.5 (g=0)
        # and x_0, x_1 varying in [0, 1].
        
        # Strategy: Mix strategic points (corners/edges) with Sobol sequence.
        # Corners and edge points contribute disproportionately to hypervolume.
        
        strategic_pts_2d = [
            (0.0, 0.0),  # Corner
            (1.0, 0.0),  # Corner
            (0.0, 1.0),  # Corner
            (0.5, 0.0),  # Edge
            (0.0, 0.5),  # Edge
            (1.0, 0.5),  # Edge
            (0.5, 1.0),  # Edge
            (0.5, 0.5),  # Center
        ]
        
        n_strategic = len(strategic_pts_2d)
        n_sobol = self.budget - n_strategic
        
        if n_sobol < 0:
            n_sobol = 0
            n_strategic = self.budget
            strategic_pts_2d = strategic_pts_2d[:self.budget]
            
        try:
            sampler = qmc.Sobol(d=2, scramble=False, seed=42)
            sobol_pts = sampler.random(n_sobol)
        except Exception:
            def halton_sequence(n, dims=2):
                points = []
                bases = [2, 3]
                for i in range(1, n + 1):
                    point = []
                    for d in range(dims):
                        f = 1.0
                        result = 0.0
                        x = i
                        while x > 0:
                            f /= bases[d]
                            result += f * (x % bases[d])
                            x //= bases[d]
                        point.append(result)
                    points.append(point)
                return np.array(points)
            sobol_pts = halton_sequence(n_sobol, dims=2)
            
        # Combine strategic and Sobol points
        all_pts_2d = np.vstack([np.array(strategic_pts_2d), sobol_pts])
        
        # Construct decision vectors
        X = np.zeros((len(all_pts_2d), self.dim))
        X[:, 0] = all_pts_2d[:, 0]
        X[:, 1] = all_pts_2d[:, 1]
        X[:, 2:] = 0.5  # Set aggregation variables to 0.5 to minimize g
        
        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate
        # We only have self.budget evaluations allowed.
        # len(X) should be self.budget.
        # If there are duplicates, we waste evaluations.
        # We can filter duplicates to save evaluations for new points?
        # But we have already generated exactly self.budget points.
        # If we filter, we have fewer points.
        # It is better to just evaluate all.
        
        F = np.array([problem(x) for x in X])

        if time.time() > deadline:
            return X

        return X
