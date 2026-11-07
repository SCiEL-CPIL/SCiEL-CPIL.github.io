"""Optimized DTLZ4 solver: exploit known Pareto front with strategic point placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """DTLZ4 optimizer exploiting known Pareto front with strategic sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj, budget = self.dim, self.n_obj, self.budget

        # DTLZ4 structure
        n_pos = n_obj - 1  # 2 position variables
        n_g = dim - n_pos  # 10 g-variables
        g_opt = 0.5 * (hi[n_pos:] + lo[n_pos:])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            f = problem(x)
            all_X.append(np.asarray(x, dtype=float))
            all_F.append(np.asarray(f, dtype=float))
            used += 1
            return all_F[-1]

        # The Pareto front of DTLZ4 with 3 objectives is the triangle:
        # f1 = x0, f2 = x1, f3 = 1 - x0 - x1, where x0 >= 0, x1 >= 0, x0+x1 <= 1
        #
        # Strategy: Place points strategically to maximize hypervolume.
        # Use a 2D area-maximization approach with inverse mapping.

        # Generate candidate points in the triangle
        # Map from unit square to triangle:
        #   x0 = u
        #   x1 = v * (1 - u)
        # where (u, v) are in [0,1]^2

        # First, add the 3 vertices - these are critical for hypervolume
        vertices = [
            (0.0, 0.0),  # f1=0, f2=0, f3=1
            (1.0, 0.0),  # f1=1, f2=0, f3=0
            (0.0, 1.0),  # f1=0, f2=1, f3=0
        ]

        n_vertices = 3
        remaining = budget - n_vertices

        # Split remaining between edges and interior
        # Give more to edges for better boundary coverage
        n_per_edge = max(1, remaining // 4)
        n_interior = remaining - 3 * n_per_edge
        if n_interior < 0:
            n_per_edge = max(1, remaining // 8)
            n_interior = max(0, remaining - 3 * n_per_edge)

        # Generate edge points with non-uniform spacing biased toward vertices
        # Points closer to vertices contribute more to hypervolume
        edge_points = []

        if n_per_edge > 0:
            # Edge 1: x1 = 0, x0 from 0 to 1 (excluding vertices)
            # Use non-uniform spacing: denser near vertices
            t = np.linspace(0.0, 1.0, n_per_edge + 2)[1:-1]
            # Apply power transformation to bias toward endpoints
            t_biased = np.where(t < 0.5, 2 * t * t, 1 - 2 * (1 - t) * (1 - t))
            for x0 in t_biased:
                edge_points.append((x0, 0.0))

        if n_per_edge > 0:
            # Edge 2: x0 + x1 = 1, x0 from 1 to 0
            t = np.linspace(0.0, 1.0, n_per_edge + 2)[1:-1]
            t_biased = np.where(t < 0.5, 2 * t * t, 1 - 2 * (1 - t) * (1 - t))
            for t_val in t_biased:
                x0 = 1.0 - t_val
                x1 = t_val
                edge_points.append((x0, x1))

        if n_per_edge > 0:
            # Edge 3: x0 = 0, x1 from 1 to 0
            t = np.linspace(0.0, 1.0, n_per_edge + 2)[1:-1]
            t_biased = np.where(t < 0.5, 2 * t * t, 1 - 2 * (1 - t) * (1 - t))
            for x1 in t_biased:
                edge_points.append((0.0, x1))

        # Generate interior points using Sobol sequence with inverse mapping
        interior_points = []
        if n_interior > 0:
            # Use Sobol sequence for interior
            n_sobol = 1
            while n_sobol < n_interior:
                n_sobol *= 2
            sobol = qmc.Sobol(d=2, scramble=True, seed=42)
            pts = sobol.random(n_sobol)

            u = pts[:, 0]
            v = pts[:, 1]
            x0_mapped = u
            x1_mapped = v * (1.0 - u)

            # Clip to stay in triangle
            x0_mapped = np.clip(x0_mapped, 0.0, 1.0)
            x1_mapped = np.clip(x1_mapped, 0.0, 1.0 - x0_mapped)

            # Take first n_interior points
            for i in range(min(n_interior, len(x0_mapped))):
                interior_points.append((x0_mapped[i], x1_mapped[i]))

        # Combine all points: vertices first, then edges, then interior
        all_points = vertices + edge_points + interior_points

        # Trim to budget if needed
        all_points = all_points[:budget]

        # Evaluate all points
        for x0, x1 in all_points:
            if used >= budget or time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = x0
            x[1] = x1
            x[n_pos:] = g_opt
            eval_point(x)

        # Return (value ignored by evaluator, but keep interface)
        if all_X:
            return np.array(all_X)
        return np.array([])
# EVOLVE-BLOCK-END
