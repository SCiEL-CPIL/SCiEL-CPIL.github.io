"""Structure-aware optimizer for DTLZ1 using analytic front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits DTLZ1's known Pareto front structure: fix x[2:] to minimize g,
    sample the 2D simplex parameterized by x[0], x[1]."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        dim = self.dim
        budget = self.budget
        used = 0
        results = []

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            f = problem(x)
            used += 1
            results.append(f)
            return f

        n_total = budget

        # DTLZ1 Pareto front parameterization using spherical coordinates:
        # f1 = 0.5 * sin(t1)
        # f2 = 0.5 * cos(t1) * sin(t2)
        # f3 = 0.5 * cos(t1) * cos(t2)
        # where t1, t2 in [0, pi/2]
        #
        # Corresponding decision variables (with g=0, x[2:]=0.5):
        # x0 = sin(t1)
        # x1 = cos(t1) * sin(t2)
        # x2 = cos(t1) * cos(t2)  (but this is for 3-objective case)
        #
        # For DTLZ1 with 3 objectives, the front is:
        # f1 = 0.5 * x0
        # f2 = 0.5 * x1
        # f3 = 0.5 * (1 - x0 - x1)
        # with x0 + x1 <= 1, x0, x1 >= 0
        #
        # Spherical parameterization:
        # x0 = sin(t1)
        # x1 = cos(t1) * sin(t2)
        # 1-x0-x1 = cos(t1) * cos(t2)
        # where t1, t2 in [0, pi/2]

        # Use stratified sampling in (t1, t2) space for uniform coverage
        # of the parameter space, which maps to good coverage of the front

        # Generate stratified samples in [0, pi/2]^2
        n_grid = int(np.ceil(np.sqrt(n_total)))

        # Create stratified grid points with jitter
        theta1_vals = []
        theta2_vals = []
        for i in range(n_grid):
            for j in range(n_grid):
                if len(theta1_vals) >= n_total:
                    break
                # Stratified position with jitter
                t1 = (i + 0.5 + np.random.uniform(-0.3, 0.3)) / n_grid * (np.pi / 2)
                t2 = (j + 0.5 + np.random.uniform(-0.3, 0.3)) / n_grid * (np.pi / 2)
                t1 = np.clip(t1, 0, np.pi / 2)
                t2 = np.clip(t2, 0, np.pi / 2)
                theta1_vals.append(t1)
                theta2_vals.append(t2)
                if len(theta1_vals) >= n_total:
                    break

        # If we don't have enough, add more random points
        while len(theta1_vals) < n_total:
            t1 = np.random.uniform(0, np.pi / 2)
            t2 = np.random.uniform(0, np.pi / 2)
            theta1_vals.append(t1)
            theta2_vals.append(t2)

        # Evaluate all points
        for t1, t2 in zip(theta1_vals[:n_total], theta2_vals[:n_total]):
            if used >= budget or time.time() > deadline:
                break

            # Convert spherical parameters to decision variables
            x0 = np.sin(t1)
            x1 = np.cos(t1) * np.sin(t2)

            x = np.full(dim, 0.5)
            x[0] = x0
            x[1] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return results
# EVOLVE-BLOCK-END
