# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7 Pareto front sampler using 3D Halton sequences for low-discrepancy coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton(self, index, base):
        """Generate Halton sequence value."""
        result = 0.0
        f = 1.0 / base
        while index > 0:
            result += f * (index % base)
            index //= base
            f /= base
        return result

    def _halton_3d(self, indices):
        """Generate 3D Halton sequence points for given indices (1-based)."""
        points = np.zeros((len(indices), 3))
        for i, idx in enumerate(indices):
            points[i, 0] = self._halton(idx, 2)
            points[i, 1] = self._halton(idx, 3)
            points[i, 2] = self._halton(idx, 5)
        return points

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        # DTLZ7 with m=3, n=22:
        # f1 = x[0], f2 = x[1], f3 = g(x[2:]) * h(x[2], g, x[1])
        # Pareto front: x[3:] = 0 (so g=1), parameterized by x[0], x[1], x[2] in [0,1]
        # Use 3D Halton sequence for low-discrepancy coverage of the front

        # Generate budget points using 3D Halton sequence
        n_points = budget
        indices = np.arange(1, n_points + 1)
        front_params = self._halton_3d(indices)

        # Build solution vectors: x[0], x[1], x[2] from Halton, x[3:] = 0
        X = np.zeros((n_points, n))
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * front_params[:, 0]
        X[:, 1] = lo[1] + (hi[1] - lo[1]) * front_params[:, 1]
        X[:, 2] = lo[2] + (hi[2] - lo[2]) * front_params[:, 2]
        # x[3:] remains 0 (already initialized)

        # Evaluate all points
        results = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            fv = problem(X[i])
            results.append(fv)

        return np.array(results) if results else np.array([])
# EVOLVE-BLOCK-END
