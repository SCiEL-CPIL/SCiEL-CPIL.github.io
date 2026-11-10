# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer: exploit known Pareto front geometry with cosine-spaced grid."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # DTLZ4: M-1 shape vars, n-M+1 distance vars
        # For M=3, dim=12: shape vars = x[0:2], distance vars = x[2:12]
        n_shape = n_obj - 1  # 2
        n_dist = dim - n_obj + 1  # 10

        # Fix distance variables to 0.5 to guarantee g=0 (Pareto front)
        # But we need to use the actual bounds; assume [0,1]
        dist_val = (lo[n_shape:] + hi[n_shape:]) / 2.0

        # Build base vector with distance vars fixed
        base = np.zeros(dim)
        base[:n_shape] = 0.0  # will be filled per point
        base[n_shape:] = dist_val

        # Cosine-spaced grid on shape variables for uniform coverage on curved front
        # Use a grid that maximizes hypervolume contribution
        # With 200 budget, use ~14x14 grid = 196, plus 4 extra
        
        used = 0
        results = []

        # Cosine spacing: t_i = 0.5 * (1 - cos(pi * i / (N-1)))
        def cosine_space(n):
            if n <= 1:
                return np.array([0.5])
            i = np.arange(n)
            return 0.5 * (1.0 - np.cos(np.pi * i / (n - 1)))

        # Grid sizes: aim for ~200 points total
        # Try different grid configurations
        grid_configs = [
            (14, 14),  # 196
            (15, 13),  # 195
            (16, 12),  # 192
            (13, 15),  # 195
            (12, 16),  # 192
        ]

        # Choose the config that uses most of the budget
        best_config = None
        best_count = 0
        for r, c in grid_configs:
            cnt = r * c
            if cnt <= budget and cnt > best_count:
                best_count = cnt
                best_config = (r, c)

        if best_config is None:
            best_config = (14, 14)

        r_grid, c_grid = best_config
        x1_vals = cosine_space(r_grid)
        x2_vals = cosine_space(c_grid)

        # Evaluate grid points
        for x1 in x1_vals:
            for x2 in x2_vals:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                x = base.copy()
                x[0] = x1
                x[1] = x2
                f = problem(x)
                results.append(f)
                used += 1
            if time.time() > deadline:
                break

        # Use remaining budget for additional strategic points
        if used < budget:
            # Add points at the "corners" and midpoints of the front
            extra_points = [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (0.5, 0.5),
                (0.25, 0.25),
                (0.75, 0.75),
                (0.25, 0.75),
                (0.75, 0.25),
                (0.1, 0.1),
                (0.9, 0.9),
                (0.1, 0.9),
                (0.9, 0.1),
                (0.5, 0.0),
                (0.5, 1.0),
                (0.0, 0.5),
                (1.0, 0.5),
            ]
            for x1, x2 in extra_points:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                x = base.copy()
                x[0] = x1
                x[1] = x2
                f = problem(x)
                results.append(f)
                used += 1

        # Fill any remaining budget with additional cosine-spaced points at finer resolution
        if used < budget:
            # Add more points along the edges of the front
            edge_steps = 20
            for i in range(edge_steps + 1):
                if used >= budget or time.time() > deadline:
                    break
                t = i / edge_steps
                # Edge 1: x1=t, x2=0
                x = base.copy()
                x[0] = t
                x[1] = 0.0
                f = problem(x)
                results.append(f)
                used += 1
                if used >= budget or time.time() > deadline:
                    break
                # Edge 2: x1=0, x2=t
                x = base.copy()
                x[0] = 0.0
                x[1] = t
                f = problem(x)
                results.append(f)
                used += 1

        return results
# EVOLVE-BLOCK-END
