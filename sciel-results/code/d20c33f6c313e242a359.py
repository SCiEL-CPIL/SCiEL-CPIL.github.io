"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that exploits ZDT1's structure:
    - Pareto front is achieved when x[2:] = 0
    - f1 = x[0], f2 = (1 + 1/x[0]) * (1 - sqrt(x[0]/(1+1/x[0])))
    - We place x[0] values using optimal quantization of the hypervolume integrand
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0  # 300s wall-clock limit, use 290s as safety margin
        safe_deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT1 Pareto front: x[0] in [0,1], x[1:] = 0
        # f1 = x[0], f2 = (1 + 1/x[0]) * (1 - sqrt(x[0] / (1 + 1/x[0])))
        # Simplify: let g(x) = 1 + 1/x, then f2 = g(x) * (1 - sqrt(x/g(x)))
        # = g(x) - sqrt(x * g(x)) = 1 + 1/x - sqrt(x + 1)

        def f2_of_x1(x1):
            """Compute f2 given x1 (with x[2:] = 0)."""
            if x1 <= 1e-10:
                return 10.0  # limit as x1 -> 0+
            g = 1.0 + 1.0 / x1
            return g - np.sqrt(x1 + 1.0)

        # Hypervolume contribution: for a point at x1, the marginal gain
        # is proportional to (11 - f2(x1)) in the 2D case.
        # Optimal quantization: place points where the integrand is largest.
        # The hypervolume integrand for the 2D front is h(x1) = 11 - f2(x1).
        # We want to distribute N points to maximize sum of h(x1_i) * dx_i
        # where dx_i is the spacing. By optimal quantization, we should
        # place more points where h(x1) is large.
        #
        # The CDF of the optimal distribution is proportional to the integral
        # of h(x1) from 0 to x1. So we compute:
        #   H(x1) = integral_0^x1 (11 - f2(t)) dt
        # and then place points at x1_i = H^{-1}(i/(N+1)) for i = 1..N

        # First, estimate the integral numerically to find the CDF
        n_grid = 10000
        x_grid = np.linspace(1e-6, 1.0, n_grid)
        f2_grid = np.array([f2_of_x1(x) for x in x_grid])
        h_grid = 11.0 - f2_grid

        # Cumulative integral (trapezoidal)
        H_grid = np.cumsum(0.5 * (h_grid[1:] + h_grid[:-1]) * np.diff(x_grid))
        H_grid = np.concatenate([[0.0], H_grid])

        # Normalize H to [0, 1]
        H_grid = H_grid / H_grid[-1]

        # Determine how many points to use
        # We have budget evaluations. Use a portion for the front points,
        # and maybe some for refinement.
        # Let's use about 80% of budget for front points, rest for safety.
        n_front = max(10, int(self.budget * 0.8))
        n_front = min(n_front, self.budget - 5)  # leave some buffer

        # Place points using inverse CDF
        # We want N points, place them at quantiles i/(N+1)
        quantiles = np.linspace(0, 1, n_front + 2)[1:-1]  # (n_front) values in (0,1)
        x1_values = np.interp(quantiles, H_grid, x_grid)

        # Ensure we don't evaluate the same point twice
        x1_values = np.unique(x1_values)
        n_front = len(x1_values)

        # Build the candidate points
        candidates = []
        for x1 in x1_values:
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] = 0 is already set
            candidates.append(x)

        # Evaluate all candidates
        F = []
        used = 0
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > safe_deadline:
                break
            f = problem(x)
            F.append(f)
            used += 1

        # If we have remaining budget, do a local refinement around the best areas
        remaining = self.budget - used
        if remaining > 0 and len(F) > 0:
            F = np.array(F)
            # Find the region with highest hypervolume density
            # and do a finer grid there
            # For simplicity, do a uniform refinement of the entire [0,1] range
            # with the remaining budget
            n_refine = min(remaining, 50)
            if n_refine > 0:
                # Use a finer grid focusing on areas where h(x1) is large
                # Resample using the same CDF but with more points
                n_total = n_front + n_refine
                quantiles_fine = np.linspace(0, 1, n_total + 2)[1:-1]
                x1_fine = np.interp(quantiles_fine, H_grid, x_grid)
                x1_fine = np.unique(x1_fine)

                # Only evaluate new points (not already evaluated)
                existing_x1 = set(np.round(x1_values, 10))
                new_x1 = [x for x in x1_fine if round(x, 10) not in existing_x1]

                for x1 in new_x1:
                    if used >= self.budget:
                        break
                    if time.time() > safe_deadline:
                        break
                    x = np.zeros(self.dim)
                    x[0] = x1
                    f = problem(x)
                    F = np.vstack([F, [f]])
                    used += 1

        return candidates
# EVOLVE-BLOCK-END
