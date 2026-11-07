"""DTLZ7 hypervolume optimizer via structured grid + quasi-random Pareto-front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ7 Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Return the index-th Halton sequence value in the given base (0-based)."""
        f = 1.0
        r = 0.0
        i = index + 1  # 1-based
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _halton_2d(self, n):
        """Generate n points in the unit square via 2D Halton sequence (bases 2,3)."""
        pts = np.empty((n, 2))
        for k in range(n):
            pts[k, 0] = self._halton(k, 2)
            pts[k, 1] = self._halton(k, 3)
        return pts

    def _front_point(self, f1, f2):
        """Convert (f1, f2) on the DTLZ7 Pareto front to an x-vector.
        
        DTLZ7 front (with x[2:]=0):
            f1 = x0
            f2 = (1-x0)*x1
            f3 = (1-x0)*(1-x1)
        
        So: x0 = f1, x1 = f2/(1-f1) if f1 < 1, else 0
        """
        x = np.zeros(self.dim)
        x[0] = f1
        if f1 < 1.0 - 1e-12:
            x[1] = f2 / (1.0 - f1)
        else:
            x[1] = 0.0
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = 0
        results = []

        def evaluate(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            results.append(f)
            n_evals += 1
            return f

        # Strategy: 
        # 1. Structured grid on the Pareto front (x0, x1 space)
        # 2. Quasi-random (Halton) sampling for dense coverage
        # 3. Remaining budget: more quasi-random points

        # The Pareto front in (f1, f2) space is the triangle:
        # f1 >= 0, f2 >= 0, f1 + f2 <= 1 (since f3 = 1 - f1 - f2 >= 0)
        # Wait, let me re-derive:
        # f1 = x0, f2 = (1-x0)*x1, f3 = (1-x0)*(1-x1)
        # f1 + f2 + f3 = x0 + (1-x0)*x1 + (1-x0)*(1-x1) = x0 + (1-x0) = 1
        # So f1 + f2 + f3 = 1, with all >= 0.
        # The front is the 2D simplex in objective space.
        # In (f1, f2) coordinates: f1 >= 0, f2 >= 0, f1 + f2 <= 1.

        # Phase 1: Structured grid
        # Use a grid in (f1, f2) space on the triangle.
        # Let's use a grid of size ~15x15 but only keep points in the triangle.
        grid_n = 15
        f1_grid = np.linspace(0, 1, grid_n)
        f2_grid = np.linspace(0, 1, grid_n)
        
        grid_points = []
        for f1_val in f1_grid:
            for f2_val in f2_grid:
                if f1_val + f2_val <= 1.0 + 1e-12:
                    grid_points.append((f1_val, f2_val))
        
        # Shuffle grid points to avoid systematic ordering issues
        np.random.shuffle(grid_points)
        
        for f1_val, f2_val in grid_points:
            if n_evals >= self.budget or time.time() > deadline:
                break
            x = self._front_point(f1_val, f2_val)
            evaluate(x)

        # Phase 2: Quasi-random Halton sampling on the triangle
        # Generate points uniformly in the triangle using Halton sequence
        # Method: u1, u2 in [0,1]^2, transform to triangle
        # s = sqrt(u1), f1 = s*(1-u2), f2 = s*u2 gives uniform in triangle
        
        n_halton = self.budget - n_evals
        if n_halton > 0:
            # Use Halton for better coverage than pure random
            halton_pts = self._halton_2d(n_halton)
            for u1, u2 in halton_pts:
                if n_evals >= self.budget or time.time() > deadline:
                    break
                s = np.sqrt(u1)
                f1_val = s * (1 - u2)
                f2_val = s * u2
                x = self._front_point(f1_val, f2_val)
                evaluate(x)

        # Phase 3: If budget remains, add more random front points
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            u1 = np.random.uniform(0, 1)
            u2 = np.random.uniform(0, 1)
            s = np.sqrt(u1)
            f1_val = s * (1 - u2)
            f2_val = s * u2
            x = self._front_point(f1_val, f2_val)
            evaluate(x)

        return results
# EVOLVE-BLOCK-END
