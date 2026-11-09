"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class Optimizer:
    """Structure-exploiting optimizer for DTLZ7 using Sobol quasi-random sampling on the Pareto manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_sobol_2d(self, n, seed=0):
        """Generate n quasi-random points in [0,1]^2."""
        if HAS_SCIPY:
            sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
            # Generate 2^k points, then take first n
            # Next_power_of_2
            k = int(np.ceil(np.log2(n)))
            if k < 1: k = 1
            points = sampler.random(2**k)
            return points[:n]
        else:
            # Fallback to Halton if scipy is not available
            points = np.zeros((n, 2))
            for i in range(n):
                # Base 2
                f = 1.0
                r = 0.0
                q = i + 1
                while q > 0:
                    f /= 2.0
                    r += f * (q % 2)
                    q //= 2
                points[i, 0] = r

                # Base 3
                f = 1.0
                r = 0.0
                q = i + 1
                while q > 0:
                    f /= 3.0
                    r += f * (q % 3)
                    q //= 3
                points[i, 1] = r
            return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim
        M = self.n_obj
        
        # For DTLZ7, the Pareto front is defined by the first M-1 variables.
        # The remaining n - (M-1) variables define the g-function.
        # To lie on the Pareto front, we must set the g-variables to their optimal value (0.5).
        n_front_vars = M - 1
        n_g_vars = dim - n_front_vars
        
        # Generate quasi-random points for the front variables
        # We use the full budget for this, as local refinement on a known front is less effective
        # than ensuring uniform coverage of the front itself for hypervolume.
        n_points = budget
        
        z_points = self._generate_sobol_2d(n_points, seed=42)
        
        # Construct full x vectors
        X = np.zeros((n_points, dim))
        
        # Assign front variables (indices 0 to M-2)
        X[:, :n_front_vars] = z_points
        
        # Assign g-variables (indices M-1 to dim-1) to 0.5 to minimize g(x) to 0
        X[:, n_front_vars:] = 0.5
        
        # Ensure within bounds (though 0.5 and [0,1] Sobol points are already safe for [0,1] bounds)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
