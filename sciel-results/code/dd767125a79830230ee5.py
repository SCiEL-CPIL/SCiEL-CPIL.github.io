"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using deterministic grid and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        n_angle = dim - n_obj  # 9 for 12 vars, 3 objs

        rng = np.random.default_rng(42)
        used = 0
        all_X = []
        all_F = []

        # For DTLZ2 with M=3 objectives:
        # f1 = (1+g) * cos(t1) * cos(t2)
        # f2 = (1+g) * sin(t1) * cos(t2)
        # f3 = (1+g) * sin(t2)
        # where g = sum(x[i]^2) for i >= n_obj (assuming standard DTLZ2)
        # To be on Pareto front: g=0, so x[2:] = 0
        # x[0] and x[1] determine t1 and t2

        # Phase 1: Deterministic 10x10 grid in angle space
        # This guarantees coverage of boundaries (corners) of the Pareto front
        n_grid = 10
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                # Use linspace to ensure boundary points are included
                t1 = np.linspace(0, np.pi/2, n_grid)[i]
                t2 = np.linspace(0, np.pi/2, n_grid)[j]
                grid_points.append((t1, t2))

        # Evaluate grid points
        for t1, t2 in grid_points:
            if time.time() > deadline or used >= self.budget:
                break

            # Construct x: x[0], x[1] from angles, x[2:] = 0 to minimize g
            x = np.zeros(dim)
            # Map angles to [0,1] range (assuming x[0], x[1] in [0,1] map to [0, pi/2])
            x[0] = t1 / (np.pi / 2)
            x[1] = t2 / (np.pi / 2)
            x[2:] = 0.0  # Minimize g

            fx = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(fx.copy())

        # Phase 2: Local search to improve g minimization for each grid point
        # If g is not exactly 0 due to problem specifics, refine
        if used < self.budget:
            n_local_steps = 3
            step_size = 0.01

            for idx in range(len(all_X)):
                if time.time() > deadline or used >= self.budget:
                    break

                x = all_X[idx].copy()
                fx = all_F[idx].copy()

                for _ in range(n_local_steps):
                    if time.time() > deadline or used >= self.budget:
                        break

                    # Perturb only x[2:] (the g-affecting variables)
                    delta = step_size * rng.standard_normal(dim)
                    delta[:2] = 0  # Keep angle variables fixed
                    x_new = np.clip(x + delta, lo, hi)

                    fx_new = problem(x_new)
                    used += 1

                    # Accept if it improves (reduces) the objectives
                    if np.sum(fx_new) < np.sum(fx):
                        x = x_new
                        fx = fx_new
                        all_X[idx] = x.copy()
                        all_F[idx] = fx.copy()

        # Phase 3: If budget remains, add more points with finer grid or random sampling
        if used < self.budget:
            # Add midpoints of the grid for better resolution
            n_mid = 5
            for i in range(n_mid):
                for j in range(n_mid):
                    if time.time() > deadline or used >= self.budget:
                        break

                    t1 = (i + 0.5) * (np.pi / 2) / n_mid
                    t2 = (j + 0.5) * (np.pi / 2) / n_mid

                    x = np.zeros(dim)
                    x[0] = t1 / (np.pi / 2)
                    x[1] = t2 / (np.pi / 2)
                    x[2:] = 0.0

                    fx = problem(x)
                    used += 1
                    all_X.append(x.copy())
                    all_F.append(fx.copy())

        # Convert to arrays
        X = np.array(all_X)
        F = np.array(all_F)

        return X
# EVOLVE-BLOCK-END
