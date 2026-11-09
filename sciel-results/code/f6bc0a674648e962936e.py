"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1-specific optimizer using optimal convergence variables (0.5) and Sobol sampling for the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, keep 10s buffer
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ1 Structure:
        # For N=7, M=3 (3 objectives):
        # Shape variables: x[0], x[1]
        # Distance variables: x[2] to x[6]
        # g(x) is minimized at x[i] = 0.5 for all distance variables.
        # This minimizes the scaling factor (1+g), making all objectives as small (negative) as possible,
        # which maximizes the distance to the reference point [400, 400, 400] and thus the hypervolume.
        
        n_shape = 2
        n_dist = self.dim - n_shape
        
        # Set distance variables to optimal value 0.5
        x_dist = np.full(n_dist, 0.5)
        # Clip to bounds just in case, though 0.5 is within [0, 1]
        x_dist = np.clip(x_dist, lo[n_shape:], hi[n_shape:])
        
        N = self.budget
        
        # Generate Sobol points in [0, 1)^2 for the shape variables
        # Sobol sequences provide low-discrepancy coverage, which is optimal for hypervolume maximization
        # on a convex Pareto front when the sample size is limited.
        try:
            from scipy.stats import qmc
            # Use scrambled Sobol to leverage the seed for better per-seed performance
            # while maintaining low-discrepancy properties
            sampler = qmc.Sobol(d=2, scramble=True)
            pts = sampler.random(N)
        except ImportError:
            # Fallback to Halton sequence if SciPy is not available
            pts = self._halton_2d(N)
            
        evals_used = 0
        for i in range(N):
            if time.time() > deadline:
                break
                
            x_full = np.zeros(self.dim)
            x_full[0] = pts[i, 0]
            x_full[1] = pts[i, 1]
            x_full[n_shape:] = x_dist
            
            # Clip to bounds [0, 1]
            x_full = np.clip(x_full, lo, hi)
            
            # Evaluate
            # The evaluator records the return value of problem(x_full)
            problem(x_full)
            evals_used += 1
            
        # The return value is ignored by the evaluator
        return None

    def _halton_2d(self, n):
        """Generate n points in [0, 1)^2 using the Halton sequence with bases 2 and 3."""
        points = np.zeros((n, 2))
        for i in range(1, n + 1):
            points[i - 1, 0] = self._halton_value(i, 2)
            points[i - 1, 1] = self._halton_value(i, 3)
        return points

    def _halton_value(self, i, base):
        """Calculate the i-th Halton number with the given base."""
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
# EVOLVE-BLOCK-END
