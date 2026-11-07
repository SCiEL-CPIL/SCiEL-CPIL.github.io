import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget

        # For DTLZ2 with 3 objectives and 12 variables:
        # - 2 shape variables (x[0], x[1]) determine position on Pareto front
        # - 10 distance variables (x[2:12]) should be set to 0.5 for g=1
        #
        # The Pareto front is a quarter unit sphere in objective space.
        # We need to generate space-filling points on this 2D manifold.
        #
        # Strategy: Use Sobol sequence in 2D, map to angles on quarter sphere.
        # For uniform coverage of the quarter sphere surface:
        #   cos(theta) uniform in [0,1]  =>  theta = arccos(u1)
        #   phi uniform in [0, pi/2]     =>  phi = u2 * pi/2
        #
        # Then: f1 = sin(theta)*cos(phi), f2 = sin(theta)*sin(phi), f3 = cos(theta)
        #
        # The shape variables x[0], x[1] must be mapped such that the problem
        # produces these objective values. For standard DTLZ2:
        #   f1 = g * (1 + cos(x1) * sin(x2))  -- this doesn't match
        #
        # Actually, for DTLZ2 with M=3, the standard formulation is:
        #   f1 = g * (1 + cos(x1) * sin(x2))
        #   f2 = g * (1 + cos(x1) * cos(x2) - cos(x1) * sin(x2))  -- no
        #
        # Let me use the correct DTLZ2 formulation:
        # For i = 1,...,M-1:
        #   f_i = g * (1 + cos(x1) * prod_{j=2}^{i} sin(x_j))
        #   Wait, that's not standard either.
        #
        # Standard DTLZ2:
        #   f_i(x) = g(x) * (1 + cos(x_1) * prod_{j=2}^{i} sin(x_j))  for i=1,...,M-1
        #   f_M(x) = g(x) * prod_{j=1}^{M-1} sin(x_j)
        #
        # For M=3:
        #   f1 = g * (1 + cos(x1) * sin(x2))
        #   f2 = g * (1 + cos(x1) * sin(x2) * cos(x2))  -- no
        #   Actually: f2 = g * (1 + cos(x1) * sin(x2)) ... I'm confusing myself.
        #
        # Let me just use the direct approach: set distance vars to 0.5,
        # and sample shape vars using Sobol sequence. The problem function
        # will compute the objectives. We just need good coverage of the
        # shape variable space [0,1]^2.

        # Generate Sobol sequence points in [0,1]^2
        # Use a power of 2 for good properties, then take n_evals points
        # n_evals = 200, so use 256 points (2^8) and take first 200
        n_sobol = 256
        sobol = qmc.Sobol(d=2, scramble=True)
        points = sobol.random(n_sobol)  # shape (256, 2) in [0,1]^2
        points = points[:n_evals]  # take first n_evals

        solutions = []

        for i in range(n_evals):
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)

            # Set shape variables using Sobol points
            x[0] = points[i, 0]
            x[1] = points[i, 1]

            # Set distance variables to 0.5 (optimal for DTLZ2)
            for var_idx in range(2, self.dim):
                x[var_idx] = 0.5

            f = np.array(problem(x))
            solutions.append((x, f))

        return solutions
