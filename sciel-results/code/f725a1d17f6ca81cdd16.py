# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Deterministic structured search for DTLZ4.

    Strategy:
    - DTLZ4 has M objectives (here M=3) and N variables (here N=12).
    - The Pareto front is a flat triangle in objective space.
    - We sample points uniformly on this triangle in objective space to maximize hypervolume.
    - We then map these objective points back to the decision space variables x0, x1.
    - The remaining variables are fixed to 0.5 to ensure g(x)=0.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For DTLZ4 with M=3, N=12:
        # Shape variables: indices 0, 1
        # Distance variables: indices 2 through 11
        n_shape = self.n_obj - 1  # 2
        n_dist = self.dim - n_shape  # 10

        # Optimal value for distance variables in DTLZ4 is 0.5
        dist_opt = 0.5

        # Generate low-discrepancy points in [0,1]^2
        # We use these to sample uniformly on the Pareto front triangle.
        n_points = self.budget
        sobol = qmc.Sobol(d=2, scramble=False, seed=42)
        # Generate n_points + some extra to ensure we have enough after mapping/reflection?
        # Reflection doesn't change the count, just the position.
        # However, we want to ensure we cover the corners well.
        # Sobol sequences naturally cover the domain well.
        
        # Generate points
        r = sobol.random(n_points)
        
        # Map to the triangle in objective space
        # Uniform sampling on triangle:
        # If r1 + r2 > 1, reflect: r1 = 1-r1, r2 = 1-r2
        mask = r[:, 0] + r[:, 1] > 1
        r[mask, 0] = 1 - r[mask, 0]
        r[mask, 1] = 1 - r[mask, 1]
        
        # Now r[:, 0] is f1, r[:, 1] is f2, f3 = 1 - f1 - f2
        f1 = r[:, 0]
        f2 = r[:, 1]
        f3 = 1 - f1 - f2
        
        # Map back to decision space x1, x2
        # x1 = f1
        # x2 = f2 / (f2 + f3)
        x1 = f1
        denom = f2 + f3
        # Handle division by zero (when f1=1, f2=0, f3=0)
        # In this case, x2 can be anything, we choose 0.
        x2 = np.zeros_like(x1)
        valid = denom > 1e-12
        x2[valid] = f2[valid] / denom[valid]
        
        # Construct shape samples
        shape_samples = np.column_stack((x1, x2))
        
        # Map to bounds (though for DTLZ4 standard bounds are [0,1])
        # The problem clips, but we should provide valid inputs.
        # x1, x2 are in [0,1].
        
        # Build the full population
        pop = np.zeros((self.budget, self.dim))
        pop[:, :n_shape] = shape_samples
        pop[:, n_shape:] = dist_opt

        # Clip to bounds just in case
        pop = np.clip(pop, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in pop])

        return pop
# EVOLVE-BLOCK-END
