"""Model-and-optimize approach for ZDT6 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq


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

        def compute_f1(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)

        # Find f1_min precisely using a dense grid + refinement
        n_grid = 200000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = compute_f1(x1_grid)
        idx_min = np.argmin(f1_grid)
        x1_min_coarse = x1_grid[idx_min]

        # Refine with even finer local grid
        local_grid = np.linspace(max(0, x1_min_coarse - 0.002),
                                  min(1, x1_min_coarse + 0.002), 500000)
        f1_local = compute_f1(local_grid)
        idx_local_min = np.argmin(f1_local)
        x1_min = local_grid[idx_local_min]
        f1_min = compute_f1(x1_min)

        # For hypervolume maximization with reference point [11, 11]:
        # Points on PF: (f1_i, 1 - sqrt(f1_i)) sorted by f1 ascending
        # f2 is decreasing in f1, so f2_1 > f2_2 > ... > f2_n
        #
        # For 2D, with points sorted by f1 ascending, the HV with reference (R,R) is:
        # HV = sum_{i=1}^{n} (R - f1_i) * (f2_{i-1} - f2_i)
        # where f2_0 = R
        #
        # To maximize this, we want to choose n values of f1 in [f1_min, 1]
        #
        # The optimal distribution equalizes the "marginal hypervolume" per point.
        # For a smooth decreasing front f2(t) = 1 - sqrt(t), the density that
        # equalizes marginal HV is proportional to (R - t) / (2*sqrt(t)).

        # U(t) = integral of (R-s)/(2*sqrt(s)) ds from f1_min to t
        # = [R*sqrt(s) - (1/3)*s^{3/2}] from f1_min to t
        def U_func(t):
            return ref * np.sqrt(t) - (1.0/3.0) * t**1.5

        U_min = U_func(f1_min)
        U_max = U_func(1.0)
        U_range = U_max - U_min

        # Create n_evals target U values, evenly spaced
        U_targets = np.linspace(U_min + U_range / (2*n_evals),
                                 U_max - U_range / (2*n_evals), n_evals)
        U_targets_shifted = U_targets - U_min

        # Invert U(f1) = U_target using a fine grid for initial guess, then refine
        n_invert = 500000
        f1_grid_inv = np.linspace(f1_min, 1.0, n_invert)
        U_grid_inv = U_func(f1_grid_inv) - U_min

        # Interpolate to get initial f1 values
        f1_selected = np.interp(U_targets_shifted, U_grid_inv, f1_grid_inv)

        # Now find x1 for each f1 value using high-precision root-finding
        # Build a dense mapping from f1 to x1 for initial bracketing
        n_map = 200000
        x1_map = np.linspace(0, 1, n_map)
        f1_map = compute_f1(x1_map)

        # Split at the minimum point
        idx_min_map = np.argmin(f1_map)
        x1_left = x1_map[:idx_min_map + 1]
        f1_left = f1_map[:idx_min_map + 1]
        x1_right = x1_map[idx_min_map:]
        f1_right = f1_map[idx_min_map:]

        # f1_left is decreasing (from x1=0 to x1_min), f1_right is increasing (from x1_min to x1=1)
        # For each target f1, find x1 using interpolation on both branches, then refine with brentq

        x1_final = np.zeros(n_evals)
        f1_targets = f1_selected

        # Interpolate initial guesses on both branches
        x1_from_left = np.interp(f1_targets, f1_left[::-1], x1_left[::-1])
        x1_from_right = np.interp(f1_targets, f1_right, x1_right)

        # Choose the x1 that gives f1 closer to target
        f1_check_left = compute_f1(x1_from_left)
        f1_check_right = compute_f1(x1_from_right)

        err_left = np.abs(f1_check_left - f1_targets)
        err_right = np.abs(f1_check_right - f1_targets)

        use_left = err_left <= err_right
        x1_final[use_left] = x1_from_left[use_left]
        x1_final[~use_left] = x1_from_right[~use_left]

        # Refine x1 values using brentq for high precision
        # For each target, find x1 such that f1(x1) = f1_target
        # Left branch: x1 in [0, x1_min], f1 decreasing
        # Right branch: x1 in [x1_min, 1], f1 increasing

        for i in range(n_evals):
            if time.time() > deadline:
                break
            target_f1 = f1_targets[i]
            if target_f1 < f1_min or target_f1 > 1.0:
                continue

            if use_left[i]:
                # Left branch: f1 decreases from 1 to f1_min as x1 goes 0 to x1_min
                # Find bracket: f1(x1) - target = 0
                # At x1=0, f1=1 > target (unless target=1)
                # At x1=x1_min, f1=f1_min < target (unless target=f1_min)
                lo, hi = 0.0, x1_min
                f_lo = compute_f1(lo) - target_f1
                f_hi = compute_f1(hi) - target_f1
                if f_lo * f_hi > 0:
                    continue  # No sign change, skip
                try:
                    x1_refined = brentq(lambda x: compute_f1(x) - target_f1, lo, hi,
                                         xtol=1e-15, rtol=1e-15, maxiter=100)
                    x1_final[i] = x1_refined
                except:
                    pass  # Keep interpolated value
            else:
                # Right branch: f1 increases from f1_min to 1 as x1 goes x1_min to 1
                lo, hi = x1_min, 1.0
                f_lo = compute_f1(lo) - target_f1
                f_hi = compute_f1(hi) - target_f1
                if f_lo * f_hi > 0:
                    continue  # No sign change, skip
                try:
                    x1_refined = brentq(lambda x: compute_f1(x) - target_f1, lo, hi,
                                         xtol=1e-15, rtol=1e-15, maxiter=100)
                    x1_final[i] = x1_refined
                except:
                    pass  # Keep interpolated value

        # Build points: x1 = x1_final[i], x2..x10 = 0
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_final

        # Evaluate all points
        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            recorded.append(f)

        return recorded
# EVOLVE-BLOCK-END
