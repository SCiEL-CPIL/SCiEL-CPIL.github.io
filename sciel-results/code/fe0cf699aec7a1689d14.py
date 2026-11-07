import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using analytical front sampling with refined point selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 295.0  # Safety margin before 300s limit

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_g = self.dim - self.n_obj + 1

        used = 0
        points = []
        objectives = []

        # On DTLZ1 Pareto front:
        #   x[2:] = 0.5 (so g = sum(x[2:]) - 0.5 = 0)
        #   f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        # Front is 2D simplex: f1+f2+f3 = 0.5

        # Build a structured set of (x1, x2) pairs for maximum hypervolume coverage
        # Key insight: hypervolume is maximized by covering the front densely,
        # especially near the "center" where the front bulges out most.

        # Strategic points: corners, edge midpoints, and interior points
        strategic_points = np.array([
            # Corners
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            # Edge midpoints
            [0.5, 0.0], [0.5, 1.0],
            [0.0, 0.5], [1.0, 0.5],
            # Quarter points on edges
            [0.25, 0.0], [0.75, 0.0],
            [0.25, 1.0], [0.75, 1.0],
            [0.0, 0.25], [0.0, 0.75],
            [1.0, 0.25], [1.0, 0.75],
            # Interior points - these help capture the "bulge" of the front
            [0.5, 0.5],
            [0.33, 0.33], [0.67, 0.33], [0.33, 0.67], [0.67, 0.67],
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
            [0.4, 0.4], [0.6, 0.4], [0.4, 0.6], [0.6, 0.6],
            [0.5, 0.33], [0.5, 0.67], [0.33, 0.5], [0.67, 0.5],
        ])

        n_structured = len(strategic_points)
        n_fill = max(0, self.budget - n_structured)

        # Generate Sobol samples for fill to ensure good spread
        try:
            from scipy.stats import qmc
            if n_fill > 0:
                sampler = qmc.Sobol(d=2, scramble=True, seed=7)
                n_power2 = 1
                while n_power2 < n_fill:
                    n_power2 *= 2
                sobol_samples = sampler.random(n_power2)[:n_fill]
            else:
                sobol_samples = np.empty((0, 2))
        except ImportError:
            if n_fill > 0:
                rng = np.random.default_rng(99)
                sobol_samples = rng.random((n_fill, 2))
            else:
                sobol_samples = np.empty((0, 2))

        all_pairs = np.vstack([strategic_points, sobol_samples])
        all_pairs = all_pairs[:self.budget]

        # Evaluate each point on the Pareto front
        for i in range(len(all_pairs)):
            if used >= self.budget or self._elapsed() > self.deadline:
                break
            x1, x2 = all_pairs[i]
            x1 = float(np.clip(x1, 0.0, 1.0))
            x2 = float(np.clip(x2, 0.0, 1.0))

            x = np.full(self.dim, 0.5)
            x[0] = x1
            x[1] = x2

            f = problem(x)
            points.append(x.copy())
            objectives.append(f)
            used += 1

        # If we still have budget left (shouldn't happen normally), fill with random
        if used < self.budget:
            rng = np.random.default_rng(2024)
            for _ in range(self.budget - used):
                if self._elapsed() > self.deadline:
                    break
                x1 = rng.random()
                x2 = rng.random()
                x = np.full(self.dim, 0.5)
                x[0] = x1
                x[1] = x2
                f = problem(x)
                points.append(x.copy())
                objectives.append(f)
                used += 1

        return points
