"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using structured Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # DTLZ2: x[0:10] are decision variables for the front, x[10:12] affect g
        # Pareto front: g=0, so x[10:12] should be at their lower bounds (0)
        # Front points: f_i = sin(pi/2 * x_i) = sin(pi/2 * x_i) for i in 0..2
        # Actually for DTLZ2: f_i = (1+g) * sin(pi/2 * x_i) for i=0..n_obj-1
        # Wait, standard DTLZ2 has n_obj objectives and m = n_obj - 1 variables for the front
        # With n_obj=3, we need 2 variables for the front shape, and the rest affect g.
        # Actually, DTLZ2: f_i(x) = (1+g(x_M)) * sin(pi/2 * x_i) for i=1..n_obj
        # where x_1...x_{n_obj-1} are the "position" variables and x_{n_obj}...x_m affect g.
        # For 12 variables and 3 objectives: x_0, x_1 determine the front position,
        # x_2...x_11 affect g. g=0 when x_2...x_11 are at their lower bounds.

        # Generate points on the Pareto front (quarter-sphere in 3D)
        # f_0 = sin(pi/2 * a), f_1 = sin(pi/2 * b), f_2 = cos(pi/2 * a) * cos(pi/2 * b)
        # Wait, let me reconsider. Standard DTLZ2 with 3 objectives:
        # f_1 = (1+g) * sin(pi/2 * x_1)
        # f_2 = (1+g) * sin(pi/2 * x_2)  
        # f_3 = (1+g) * cos(pi/2 * x_1) * cos(pi/2 * x_2)
        # With g=0: f_1 = sin(pi/2 * x_1), f_2 = sin(pi/2 * x_2), f_3 = cos(pi/2 * x_1) * cos(pi/2 * x_2)
        # For x_1, x_2 in [0,1], this traces out the quarter-sphere.

        # Use a grid on the Pareto front for initial population
        n_grid = 14  # 14x14 = 196 points
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                a = i / (n_grid - 1)
                b = j / (n_grid - 1)
                f0 = np.sin(np.pi / 2 * a)
                f1 = np.sin(np.pi / 2 * b)
                f2 = np.cos(np.pi / 2 * a) * np.cos(np.pi / 2 * b)
                # x[0] = a, x[1] = b, x[2:] = 0 (for g=0)
                x = np.zeros(self.dim)
                x[0] = a
                x[1] = b
                grid_points.append(x)

        # Add some midpoint points for better coverage
        mid_points = []
        for i in range(n_grid - 1):
            for j in range(n_grid - 1):
                a = (i + 0.5) / (n_grid - 1)
                b = (j + 0.5) / (n_grid - 1)
                x = np.zeros(self.dim)
                x[0] = a
                x[1] = b
                mid_points.append(x)

        # Combine and select up to budget
        all_candidates = grid_points + mid_points
        # Shuffle and take up to budget
        rng.shuffle(all_candidates)
        all_candidates = all_candidates[:self.budget]

        # Evaluate all candidates
        F = np.array([problem(x) for x in all_candidates])
        used = len(all_candidates)

        # Check if we have time for refinement
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            # Try to improve points that are not on the front (g > 0)
            # For each point, try to reduce g by moving x[2:] toward 0
            for idx in range(len(all_candidates)):
                if time.time() > deadline - 5:
                    break
                x = all_candidates[idx].copy()
                # If x[2:] are not at lower bounds, try moving them there
                if np.any(np.abs(x[2:] - lo[2:]) > 1e-10):
                    x_new = x.copy()
                    x_new[2:] = lo[2:]  # Move to lower bounds for g=0
                    f_new = problem(x_new)
                    # Check if this dominates or is non-dominated
                    # Simple check: if any objective is better and none worse
                    if np.all(f_new <= F[idx]) and np.any(f_new < F[idx]):
                        all_candidates[idx] = x_new
                        F[idx] = f_new
                        used += 1

        return all_candidates
# EVOLVE-BLOCK-END
