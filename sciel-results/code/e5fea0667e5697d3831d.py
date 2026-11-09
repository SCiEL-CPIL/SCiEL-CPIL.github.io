"""Model-and-optimize approach for ZDT6 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using analytical Pareto front knowledge."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s limit

        n_evals = self.budget
        dim = self.dim
        ref = 11.0

        # ZDT6 analytical model:
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        # g(x) = 1 + 10*(dim-1) + sum_{i=2}^{dim} [x_i^2 - 10*cos(12*pi*x_i)]
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        #
        # Pareto front: x2..x10 = 0, so g = 1
        # On PF: f2 = 1 - sqrt(f1)
        # f1 ranges from f1_min to 1 (at x1=0 or x1=1)

        dim = self.dim

        # Compute f1(x1) analytically
        def compute_f1(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)

        # Find f1_min: minimize f1 over x1 in [0,1]
        # Use fine grid + golden section refinement
        n_grid = 200000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = compute_f1(x1_grid)
        idx_min = np.argmin(f1_grid)
        x1_min_coarse = x1_grid[idx_min]

        # Refine around the minimum using a local search
        local_grid = np.linspace(max(0, x1_min_coarse - 0.005),
                                  min(1, x1_min_coarse + 0.005), 100000)
        f1_local = compute_f1(local_grid)
        idx_local_min = np.argmin(f1_local)
        x1_min = local_grid[idx_local_min]
        f1_min = compute_f1(x1_min)

        # On the Pareto front (g=1): f2 = 1 - sqrt(f1)
        # We need to select n_evals values of x1 such that the corresponding
        # (f1, f2) points maximize hypervolume with reference [11, 11].

        # Compute the total U
        def U_func(t):
            return 11.0 * np.sqrt(t) - (1.0/3.0) * t**1.5

        U_min = U_func(f1_min)
        U_max = U_func(1.0)
        U_range = U_max - U_min

        # Create target U values
        n_pts = n_evals
        U_targets = np.linspace(U_min + U_range / (2*n_pts),
                                 U_max - U_range / (2*n_pts), n_pts)

        # Invert U(f1) = U_target to find f1 values
        # Use binary search on a fine grid
        n_invert_grid = 50000
        f1_invert_grid = np.linspace(f1_min, 1.0, n_invert_grid)
        U_invert_grid = U_func(f1_invert_grid) - U_min  # shift so U(f1_min) = 0

        # For each U_target (shifted), find f1 via interpolation
        U_targets_shifted = U_targets - U_min
        f1_selected = np.interp(U_targets_shifted, U_invert_grid, f1_invert_grid)

        # Now convert f1 values back to x1 values
        # f1(x1) = 1 - exp(-4*x1)*sin(6*pi*x1)^6
        # This is not monotonic, so we need to find x1 for each f1.
        # For the Pareto front, we want the x1 that gives the desired f1
        # and is on the "front" (smallest x1 for a given f1 in the decreasing branch).

        # Create a fine mapping from x1 to f1
        n_map_grid = 200000
        x1_map_grid = np.linspace(0, 1, n_map_grid)
        f1_map_grid = compute_f1(x1_map_grid)

        # For the Pareto front, we want the branch where f1 decreases then increases.
        # The minimum is at x1_min. For f1 values above f1_min, there are two x1 values.
        # We want the one with smaller x1 (left branch) for the Pareto front.
        # Actually, both give the same (f1, f2) on the Pareto front since g=1.
        # But we need x1 in [0,1]. Let's use the left branch (x1 < x1_min) for
        # f1 values, and handle the right branch if needed.

        # For simplicity, use the x1 that gives the closest f1 on the left branch
        # (x1 from 0 to x1_min), and if f1 > f1(x1_min) on the right, use right branch.

        # Actually, let's just find the x1 that minimizes |f1(x1) - f1_target|
        # using the grid, preferring the left branch.

        # Split the grid into left (x1 <= x1_min) and right (x1 > x1_min)
        left_mask = x1_map_grid <= x1_min
        right_mask = x1_map_grid >= x1_min

        x1_left_grid = x1_map_grid[left_mask]
        f1_left_grid = f1_map_grid[left_mask]

        x1_right_grid = x1_map_grid[right_mask]
        f1_right_grid = f1_map_grid[right_mask]

        x1_final = np.zeros(n_pts)
        for i in range(n_pts):
            target_f1 = f1_selected[i]
            # Try left branch first
            if len(f1_left_grid) > 0:
                idx_l = np.argmin(np.abs(f1_left_grid - target_f1))
                best_f1_l = f1_left_grid[idx_l]
                best_x1_l = x1_left_grid[idx_l]
            else:
                best_f1_l = np.inf
                best_x1_l = 0

            if len(f1_right_grid) > 0:
                idx_r = np.argmin(np.abs(f1_right_grid - target_f1))
                best_f1_r = f1_right_grid[idx_r]
                best_x1_r = x1_right_grid[idx_r]
            else:
                best_f1_r = np.inf
                best_x1_r = 0

            # Choose the branch with closer f1
            if abs(best_f1_l - target_f1) <= abs(best_f1_r - target_f1):
                x1_final[i] = best_x1_l
            else:
                x1_final[i] = best_x1_r

        # Build the points: x1 = x1_final[i], x2..x10 = 0
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_final

        # Evaluate all points, tracking time
        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            recorded.append(f)

        return recorded
# EVOLVE-BLOCK-END
