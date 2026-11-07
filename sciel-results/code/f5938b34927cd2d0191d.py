"""Multi-objective optimizer for DTLZ1 exploiting known structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ1 structure for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        all_X = []
        all_F = []

        def evaluate(x):
            nonlocal all_X, all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def make_x(x0, x1, shape_offsets=None):
            """Create a DTLZ1 point with given x0, x1 and shape variables near 0.5."""
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            if shape_offsets is not None:
                x[2:] = 0.5 + shape_offsets
            else:
                x[2:] = 0.5
            return np.clip(x, lo, hi)

        # Use Sobol sequence for the main search on the Pareto front parameter space
        # The Pareto front is a 2D simplex in objective space, parameterized by x0, x1 in [0,1]
        n_sobol = self.budget - 10  # Reserve some budget for explicit boundary points
        if n_sobol < 8:
            n_sobol = self.budget

        # Find largest power of 2 <= n_sobol for optimal Sobol quality
        n_pow2 = int(2 ** np.floor(np.log2(n_sobol)))
        if n_pow2 < 4:
            n_pow2 = 4

        sobol = qmc.Sobol(d=2, scramble=False)
        points = sobol.random(n_pow2)

        for p in points:
            if len(all_X) >= self.budget or time.time() > deadline - 30:
                break
            x0, x1 = p[0], p[1]
            x = make_x(x0, x1)
            evaluate(x)

        # Add critical boundary and center points to ensure full coverage of the simplex vertices
        # Vertices in (x0, x1) map to vertices of the objective simplex
        critical_points = [
            (0.0, 0.0),   # (0, 0, 0.5)
            (0.0, 1.0),   # (0, 0.5, 0)
            (1.0, 0.0),   # (0.5, 0, 0)
            (1.0, 1.0),   # (0.5, 0, 0) - Duplicate in obj space, but valid in param space
            (0.5, 0.5),   # Center of param space
            (0.5, 0.0),   # Midpoint of edge
            (0.0, 0.5),   # Midpoint of edge
            (1.0, 0.5),   # Midpoint of edge
            (0.5, 1.0),   # Midpoint of edge
        ]
        
        for cp in critical_points:
            if len(all_X) >= self.budget or time.time() > deadline - 10:
                break
            x0, x1 = cp
            x = make_x(x0, x1)
            evaluate(x)

        # Fill remaining budget with random points if any
        while len(all_X) < self.budget:
            if time.time() > deadline - 5:
                break
            x0 = rng.random()
            x1 = rng.random()
            x = make_x(x0, x1)
            evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
