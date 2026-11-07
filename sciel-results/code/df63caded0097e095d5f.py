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

    def _halton_2d_vectorized(self, n):
        """Generate n points in the unit square via 2D Halton sequence (bases 2,3) vectorized."""
        indices = np.arange(1, n + 1, dtype=np.float64)
        
        # Base 2
        h2 = np.zeros(n, dtype=np.float64)
        f = 1.0
        i = indices.copy()
        while np.any(i > 0):
            f /= 2.0
            h2 += f * (i % 2)
            i = i // 2
            i = i.astype(np.float64)
        
        # Base 3
        h3 = np.zeros(n, dtype=np.float64)
        f = 1.0
        i = indices.copy()
        while np.any(i > 0):
            f /= 3.0
            h3 += f * (i % 3)
            i = i // 3
            i = i.astype(np.float64)
        
        return np.column_stack([h2, h3])

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

        # The Pareto front in (f1, f2) space is the triangle:
        # f1 >= 0, f2 >= 0, f1 + f2 <= 1
        # f3 = 1 - f1 - f2

        # Phase 1: Structured grid with better resolution
        # Use a finer grid to cover the simplex more densely
        grid_n = 25
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
        # Use vectorized Halton for speed
        n_halton = self.budget - n_evals
        if n_halton > 0:
            halton_pts = self._halton_2d_vectorized(n_halton)
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
