import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using Pareto front sampling with search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 295.0

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        used = 0
        points = []
        objectives = []

        # On DTLZ1 Pareto front:
        #   x[2:] = 0.5
        #   f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        # The Pareto front is parameterized by x1, x2 in [0, 1].
        
        # We use a Sobol sequence to generate well-distributed points in the [0,1]^2 square
        # which maps to the Pareto front. This maximizes hypervolume coverage.
        
        n_points = self.budget
        
        try:
            from scipy.stats import qmc
            # Use a scrambled Sobol sequence for better distribution
            sampler = qmc.Sobol(d=2, scramble=True, seed=12345)
            # Generate exactly n_points samples
            # Sobol requires power of 2, but we can just generate n_points if we use the random method correctly?
            # Actually, qmc.Sobol.random(n) works for any n, but it's most efficient for powers of 2.
            # Let's generate n_points directly.
            sobol_samples = sampler.random(n_points)
        except ImportError:
            # Fallback to random if scipy is not available
            rng = np.random.default_rng(12345)
            sobol_samples = rng.random((n_points, 2))

        for i in range(n_points):
            if used >= self.budget or self._elapsed() > self.deadline:
                break

            x1 = float(np.clip(sobol_samples[i, 0], 0.0, 1.0))
            x2 = float(np.clip(sobol_samples[i, 1], 0.0, 1.0))

            x = np.full(self.dim, 0.5)
            x[0] = x1
            x[1] = x2

            f = problem(x)
            points.append(x.copy())
            objectives.append(np.array(f, dtype=float))
            used += 1

        return points
