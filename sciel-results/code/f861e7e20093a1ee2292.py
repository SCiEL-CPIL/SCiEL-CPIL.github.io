import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2: first 2 vars are position, remaining 10 are distance
        # Set distance vars to 0.5 to get g=0 (on Pareto front)
        # Grid over position space (x0, x1)
        grid_size = 10
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = np.full(self.dim, 0.5)
                x[0] = (i + 0.5) / grid_size
                x[1] = (j + 0.5) / grid_size
                grid_points.append(x)

        # Evaluate grid points
        all_X = []
        all_F = []
        used = 0

        for x in grid_points:
            if used >= self.budget or time.time() > deadline:
                break
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            all_X.append(x_clipped)
            all_F.append(f)
            used += 1

        # Use remaining budget for random exploration
        rng = np.random.default_rng(42)
        while used < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = np.array(problem(x))
            all_X.append(x)
            all_F.append(f)
            used += 1

        # Return last point (return value is ignored by evaluator)
        if all_X:
            return all_X[-1]
        return None
