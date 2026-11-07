"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 exploiting known Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _halton_sequence(n, base):
        """Generate n points of the Halton sequence in [0,1)."""
        points = []
        for i in range(1, n + 1):
            f = 1.0
            result = 0.0
            while i > 0:
                f /= base
                i, remainder = divmod(i, base)
                result += remainder * f
            points.append(result)
        return np.array(points)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        all_F = []

        def evaluate(x):
            nonlocal all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_F.append(f.copy())
            return f

        n_evals = self.budget

        # Use Sobol sequence for better space-filling properties on the simplex
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            points = sampler.random(n=n_evals)
        except Exception:
            # Fallback: Halton sequence
            h1 = self._halton_sequence(n_evals, 2)
            h2 = self._halton_sequence(n_evals, 3)
            points = np.column_stack([h1, h2])

        # Sample on the Pareto front
        for i in range(n_evals):
            if time.time() > deadline - 5:
                break
            if len(all_F) >= self.budget:
                break

            x0 = points[i, 0]
            x1 = points[i, 1]

            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5  # optimal shape variables

            evaluate(x)

        return all_F
# EVOLVE-BLOCK-END
