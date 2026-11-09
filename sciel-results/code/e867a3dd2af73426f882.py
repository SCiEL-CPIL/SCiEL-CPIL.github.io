"""Improved optimizer for DTLZ1 exploiting known Pareto front structure."""
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for DTLZ1 using analytic front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        evaluated = 0
        recorded_F = []

        # DTLZ1 structure:
        # f0 = 0.5 * x[0] * (1 + g)
        # f1 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
        # f2 = 0.5 * x[0] * x[1] * (1 + g)
        # where g depends on x[2:]
        #
        # To minimize all objectives, we need:
        # 1. Minimize g by optimizing x[2:]
        # 2. Sample x[0], x[1] to cover the Pareto front

        # Step 1: Find g-minimizing configuration for x[2:]
        # Use a local search approach with multiple restarts
        rng = np.random.default_rng(42)

        best_g_val = np.inf
        best_x2 = None

        # Multiple random restarts for g minimization
        n_restarts = 20
        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                break

            # Initialize x[2:] randomly
            x2 = rng.random(dim - 2)

            # Local search on x[2:]
            for it in range(50):
                if time.time() > deadline - 30:
                    break

                x_test = np.concatenate([[0.5, 0.5], x2])
                x_test = np.clip(x_test, lo, hi)
                f = problem(x_test)
                evaluated += 1
                g_val = np.sum(f)

                if g_val < best_g_val:
                    best_g_val = g_val
                    best_x2 = x2.copy()

                # Perturb x[2:]
                step_size = 0.1 * (1 - it / 50)
                x2_new = x2 + step_size * (hi[2:] - lo[2:]) * rng.standard_normal(dim - 2)
                x2 = np.clip(x2_new, lo[2:], hi[2:])

        # If we haven't found a good x2, try some structured candidates
        if best_x2 is None:
            candidates = [
                np.zeros(dim - 2),
                np.ones(dim - 2),
                np.full(dim - 2, 0.5),
            ]
            for x2 in candidates:
                if time.time() > deadline - 10:
                    break
                x_test = np.concatenate([[0.5, 0.5], x2])
                x_test = np.clip(x_test, lo, hi)
                f = problem(x_test)
                evaluated += 1
                g_val = np.sum(f)
                if g_val < best_g_val:
                    best_g_val = g_val
                    best_x2 = x2.copy()

        if best_x2 is None:
            best_x2 = np.zeros(dim - 2)

        # Step 2: Sample points on the Pareto front
        # The front is parameterized by x[0] in [0,1] and x[1] in [0,1]
        # We want to cover the simplex uniformly to maximize hypervolume

        # Use a stratified sampling approach for better coverage
        # Create a grid of (x0, x1) pairs that covers the simplex well
        # We need x0 + x1 <= 1 for valid simplex points
        n_grid = 12  # 12x12 = 144 points, leaving ~56 for refinement

        x0_vals = np.linspace(0.05, 0.95, n_grid)
        x1_vals = np.linspace(0.05, 0.95, n_grid)

        for x0 in x0_vals:
            if evaluated >= self.budget or time.time() > deadline:
                break
            for x1 in x1_vals:
                if evaluated >= self.budget or time.time() > deadline:
                    break
                # Ensure x0 + x1 <= 1 for valid simplex points (though DTLZ1 allows any x0, x1 in [0,1])
                # But for DTLZ1, the Pareto front is a simplex, so we need x0 + x1 <= 1
                if x0 + x1 > 1.0:
                    continue
                x = np.concatenate([[x0, x1], best_x2])
                x = np.clip(x, lo, hi)
                f = problem(x)
                evaluated += 1
                recorded_F.append(f)

        # Step 3: Use remaining budget for refinement
        remaining = self.budget - evaluated
        if remaining > 0 and recorded_F:
            F_arr = np.array(recorded_F)

            # Find non-dominated points
            n_points = len(F_arr)
            dominated = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                if dominated[i]:
                    continue
                for j in range(n_points):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break

            non_dom_idx = np.where(~dominated)[0]

            # Refine around non-dominated points with smaller steps
            n_refine = min(remaining, 20)
            for _ in range(n_refine):
                if evaluated >= self.budget or time.time() > deadline:
                    break

                if len(non_dom_idx) > 0:
                    idx = non_dom_idx[rng.integers(len(non_dom_idx))]
                    # Reconstruct x from index
                    row = idx // n_grid
                    col = idx % n_grid
                    x_base = np.concatenate([[x0_vals[row], x1_vals[col]], best_x2])
                else:
                    x_base = np.concatenate([[0.5, 0.5], best_x2])

                # Small perturbation
                step = 0.02 * (hi - lo)
                x = x_base + step * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                f = problem(x)
                evaluated += 1
                recorded_F.append(f)

        return recorded_F
